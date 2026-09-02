#!/usr/bin/env python3
"""
Phase 3 - Benchmarking 

Benchmarks pretrained VLMs on a sensitivity-labeled dataset (images tagged
with a CSS level 1-4 = "Content Sensitivity Score", 1 = least
sensitive/easiest, 4 = most sensitive/hardest) and reports THREE tables --
one each for REAL images, FAKE images (fully synthetic / AI generated), and
TAMPERED images (partially manipulated, e.g. face-swap / face attribute
edits / text-image mismatch from DGM4).

Every registered model is a VLM (Qwen-2.5VL, InternVL) prompted to
answer 'real', 'fake', or 'tampered' and scored against the sample's exact
label on all three tables -- a tampered image predicted 'fake' is WRONG,
since these models are expected to tell the two apart.

Each table has one row per VLM, one column per CSS level (1-4) with the
accuracy for that (subset, CSS level) cell, and TWO final "caAcc" (Content
Aware Accuracy) columns (both a weighted sum of the four per-level
accuracies, weighted TOWARD higher CSS levels): 
    - caAcc_linear: weights CSS levels 1,2,3,4 directly (proportional).
    - caAcc_fib:    weights CSS levels using Fibonacci numbers 1,2,3,5
                    instead of 1,2,3,4.

All three tables are printed to a log file and written to CSV
(<prefix>_real.csv, <prefix>_fake.csv, <prefix>_tampered.csv).

--------------------------------------------------------------------------
DATASET FORMAT
--------------------------------------------------------------------------
--manifest points to a JSON file shaped like manifest_css.py: a JSON array
of objects, each with (at minimum):

    {
      "image_id": "...",
      "filename": "rrdataset_fake_00.jpg",
      "source_dataset": "rrdataset",           # rrdataset | dgm4 | sidset | ...
      "label": "real" | "fake" | "tampered",
      "final_css_score": 1-4,                  # falls back to "css_score" if missing
      ...
    }

A simpler generic CSV (image_path,label,css_level) is also supported as a
fallback for other datasets -- it's auto-detected from the file extension.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Config: CSS weighting for Content Aware Accuracy (caAcc) 
# Both normalized to sum to 1.0 so they stay on a 0-100 scale, directly
# comparable to the plain accuracy columns and to each other.
# ---------------------------------------------------------------------------
_CSS_LINEAR = [1, 2, 3, 4]  # one per CSS level 1-4, in order
_CSS_FIBONACCI = [1, 2, 3, 5]  # one per CSS level 1-4, in order
CSS_WEIGHTS_LINEAR: Dict[int, float] = {
    level: raw / sum(_CSS_LINEAR) for level, raw in zip([1, 2, 3, 4], _CSS_LINEAR)
}
CSS_WEIGHTS_FIBONACCI: Dict[int, float] = {
    level: raw / sum(_CSS_FIBONACCI) for level, raw in zip([1, 2, 3, 4], _CSS_FIBONACCI)
}
CSS_LEVELS = [1, 2, 3, 4]

# The three tables we report, in print/CSV order.
SUBSETS = ["real", "fake", "tampered"]
SUBSET_TITLES = {
    "real": "Table 1: REAL images",
    "fake": "Table 2: FAKE images (fully synthetic)",
    "tampered": "Table 3: TAMPERED images (partially manipulated)",
}

REAL_LABELS = {"real", "0", "authentic", "pristine"}
FAKE_LABELS = {"fake", "1", "deepfake", "generated", "synthetic", "fully_synthetic"}
TAMPERED_LABELS = {"tampered", "manipulated", "partial_fake", "partially_manipulated"}


def normalize_label(label: str) -> str:
    l = str(label).strip().lower()
    if l in REAL_LABELS:
        return "real"
    if l in FAKE_LABELS:
        return "fake"
    if l in TAMPERED_LABELS:
        return "tampered"
    raise ValueError(f"Unrecognized ground-truth label: {label!r}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    image_id: str
    filename: str
    image_path: str
    source_dataset: str
    label: str        # "real" | "fake" | "tampered"
    css_level: int     # 1-4
    scenario: Optional[str] = None
    generator_model: Optional[str] = None


def resolve_image_path(
    images_dir: Optional[str],
    filename: str,
    source_dataset: str,
    label: str,
    path_template: Optional[str] = None,
) -> str:
    if path_template:
        return path_template.format(filename=filename, source_dataset=source_dataset, label=label)
    if not images_dir:
        return filename
    candidates = [
        Path(images_dir) / filename,
        Path(images_dir) / source_dataset / filename,
        Path(images_dir) / label / filename,
        Path(images_dir) / source_dataset / label / filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def load_manifest(
    manifest_path: str,
    images_dir: Optional[str] = None,
    path_template: Optional[str] = None,
    col_image: str = "image_path",
    col_label: str = "label",
    col_css: str = "css_level",
) -> List[Sample]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    if path.suffix.lower() in (".json", ".py"):
        with open(path) as f:
            raw = json.load(f)
    else:
        with open(path, newline="") as f:
            raw = list(csv.DictReader(f))

    samples: List[Sample] = []

    if raw and isinstance(raw[0], dict) and ("final_css_score" in raw[0] or "css_score" in raw[0]):
        # Rich manifest format (manifest_css.py): real / fake / tampered,
        # per-image sensitivity scoring metadata, etc.
        for entry in raw:
            css = entry.get("final_css_score")
            if css is None:
                css = entry.get("css_score")
            if css is None:
                continue  # unscored row -- skip rather than crash
            css = int(css)
            if css not in CSS_LEVELS:
                raise ValueError(f"css level must be 1-4, got {css} for {entry.get('filename')}")
            label = normalize_label(entry["label"])
            filename = entry["filename"]
            source_dataset = entry.get("source_dataset", "")
            img_path = resolve_image_path(images_dir, filename, source_dataset, label, path_template)
            samples.append(
                Sample(
                    image_id=entry.get("image_id", filename),
                    filename=filename,
                    image_path=img_path,
                    source_dataset=source_dataset,
                    label=label,
                    css_level=css,
                    scenario=entry.get("scenario"),
                    generator_model=entry.get("generator_model"),
                )
            )
    else:
        # Generic fallback manifest: image_path,label,css_level
        for row in raw:
            img = row[col_image]
            if images_dir and not os.path.isabs(img):
                img = str(Path(images_dir) / img)
            css = int(row[col_css])
            if css not in CSS_LEVELS:
                raise ValueError(f"css_level must be 1-4, got {css} for {img}")
            samples.append(
                Sample(
                    image_id=row.get("image_id", img),
                    filename=Path(img).name,
                    image_path=img,
                    source_dataset=row.get("source_dataset", ""),
                    label=normalize_label(row[col_label]),
                    css_level=css,
                )
            )

    if not samples:
        raise ValueError("Manifest loaded but contains 0 usable rows.")
    return samples


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------
class BaseModel(ABC):
    """Common interface every VLM must implement.

    predict() must return "real", "fake", or "tampered" for a single image
    -- every registered model here is a VLM (Qwen-2.5VL, Qwen3-VL, InternVL)
    prompted to actually distinguish fully-synthetic fakes from partially
    manipulated (tampered) images, and is scored against the sample's exact
    label on all three tables.
    """

    name: str = "BaseModel"

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda", model_name: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.device = device
        if model_name:
            # Lets one class be registered under several MODEL_REGISTRY keys
            # (e.g. QwenVLModel backs both "Qwen-2.5VL-7B" and
            # "Qwen-2.5VL-32B") while DEFAULT_CHECKPOINTS lookups and error
            # messages still show the specific registry name being run.
            self.name = model_name
        self._loaded = False

    def load(self) -> None:
        """Load weights into memory. Called once before predict() loop."""
        self._loaded = True

    def unload(self) -> None:
        """Free GPU memory between models."""
        self._loaded = False

    @abstractmethod
    def predict(self, image_path: str) -> str:
        ...


def _assert_fully_loaded(model, model_name: str) -> None:
    """device_map="auto" (used by every VLM class below) silently leaves
    some layers on the 'meta' device -- unmaterialized placeholders, not
    real weights -- when the GPUs currently visible (via --gpu) don't have
    enough combined VRAM to hold the model. 
    """
    meta_params = 0
    total_params = 0
    meta_names: List[str] = []
    for name, p in model.named_parameters():
        total_params += 1
        if p.device.type == "meta":
            meta_params += 1
            meta_names.append(name)
    if meta_params:
        preview = ", ".join(meta_names[:20])
        if len(meta_names) > 20:
            preview += f", ... ({len(meta_names) - 20} more)"
        logging.warning(f"{model_name}: parameters left on 'meta' device: {preview}")
        raise RuntimeError(
            f"{model_name}: {meta_params}/{total_params} parameters ended up on the 'meta' device "
            "-- the GPU(s) currently visible via --gpu don't have enough combined VRAM to hold this "
            "model. Free up more GPUs and widen --gpu to include them (e.g. --gpu 0,1,2,3), or use a "
            "smaller/quantized checkpoint instead."
        )


def parse_real_fake_tampered(text: str) -> str:
    """Three-way parser for the VLMs' free-text answers. Checks "tampered"
    before "fake" since a model might reasonably say something like "this is
    a tampered image" or "the face was swapped, making this fake" -- the
    tampered/manipulated language should win in either case.
    """
    t = text.strip().lower()
    if "tamper" in t or "manipulat" in t:
        return "tampered"
    if "fake" in t:
        return "fake"
    if "real" in t:
        return "real"
    # Model refused/hedged -- default to "real" rather than crashing the
    # run; this counts as a miss whenever the ground truth isn't "real".
    return "real"



_MIN_IMAGE_SIDE = 32  # comfortably above Qwen's 28px patch size and InternVL's tiling minimum


def load_and_sanitize_image(image_path: str):
    from PIL import Image, ImageOps

    img = Image.open(image_path)
    img.load()  
    img = ImageOps.exif_transpose(img)  
    if img.mode != "RGB":
        img = img.convert("RGB")
    if min(img.size) < _MIN_IMAGE_SIDE:
        scale = _MIN_IMAGE_SIDE / min(img.size)
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.BICUBIC)
    return img



# Default checkpoints -- these are HF Hub repo ids, not local directories.
DEFAULT_CHECKPOINTS: Dict[str, str] = {
    "Qwen-2.5VL-7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    "InternVL3-8B": "OpenGVLab/InternVL3-8B",
}



QWEN_PROMPT = (
    "You are a deepfake detection expert. Look at this image carefully and "
    "classify it into exactly ONE of three categories:\n"
    "- 'real': an authentic, unaltered photograph.\n"
    "- 'fake': a fully AI-generated / synthetic image (e.g. produced by a "
    "GAN or diffusion model), not derived from a real photograph.\n"
    "- 'tampered': a real photograph that has been partially manipulated or "
    "edited (e.g. a face swap, a face attribute edit, or an image edited to "
    "mismatch its caption/context).\n"
    "Respond with exactly one word: 'real', 'fake', or 'tampered'."
)


class QwenVLModel(BaseModel):

    name = "QwenVLModel"

    def load(self) -> None:
        checkpoint = self.checkpoint_path or DEFAULT_CHECKPOINTS.get(self.name)
        if not checkpoint:
            raise ValueError(f"{self.name}: pass --checkpoint {self.name}=<hf repo id or local path>")
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        # device_map="auto" (not the --device flag) to match how this model
        # was loaded for CSS scoring -- accelerate spreads it across
        # whatever GPUs are visible.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint,
            # dtype=, not the deprecated torch_dtype= -- avoids the
            # "`torch_dtype` is deprecated! Use `dtype` instead!" warning.
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        _assert_fully_loaded(self.model, self.name)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self._loaded = True

    def unload(self) -> None:
        if self._loaded:
            del self.model
            import gc

            gc.collect()
            self.torch.cuda.empty_cache()
        self._loaded = False

    def predict(self, image_path: str) -> str:
        from qwen_vl_utils import process_vision_info

        image = load_and_sanitize_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QWEN_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with self.torch.no_grad():
            # do_sample=False -- greedy decoding, not the checkpoint's own
            # default generation_config
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=8, do_sample=False, repetition_penalty=1.0
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return parse_real_fake_tampered(output_text)



INTERNVL_PROMPT = (
    "<image>\nYou are a deepfake detection expert. Classify this image into "
    "exactly ONE of three categories:\n"
    "- 'real': an authentic, unaltered photograph.\n"
    "- 'fake': a fully AI-generated / synthetic image (e.g. produced by a "
    "GAN or diffusion model), not derived from a real photograph.\n"
    "- 'tampered': a real photograph that has been partially manipulated or "
    "edited (e.g. a face swap, a face attribute edit, or an image edited to "
    "mismatch its caption/context).\n"
    "Respond with exactly one word: 'real', 'fake', or 'tampered'."
)


def _internvl_build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_ratio = _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _internvl_load_image(image_path, input_size=448, max_num=12):
    # load_and_sanitize_image() covers RGB conversion (which this already
    # did) plus EXIF-orientation normalization and a minimum-dimension floor
    image = load_and_sanitize_image(image_path)
    transform = _internvl_build_transform(input_size=input_size)
    images = _internvl_dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return [transform(img) for img in images]


def split_model_across_gpus(model_name: str) -> Dict[str, int]:
    """Multi-GPU device_map for InternVL"""
    import math
    import torch
    from transformers import AutoConfig

    device_map: Dict[str, int] = {}
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        return {"": 0}

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers

    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, n in enumerate(num_layers_per_gpu):
        for _ in range(n):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1

    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    return device_map


def _patch_tied_weights_compat() -> None:
    """Works around AttributeError: 'InternVLChatModel' object has no
    attribute 'all_tied_weights_keys' on transformers 5.13.1 (and likely
    other recent 5.x releases).
    """
    import transformers

    if getattr(transformers.PreTrainedModel, "_benchmark_tied_weights_patched", False):
        return

    _BACKING_ATTR = "_benchmark_all_tied_weights_keys"

    def _getter(self):
        if hasattr(self, _BACKING_ATTR):
            return getattr(self, _BACKING_ATTR)
        tied_keys = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied_keys, dict):
            fallback = tied_keys
        elif tied_keys:
            fallback = {k: k for k in tied_keys}
        else:
            fallback = {}
        return fallback

    def _setter(self, value):
        object.__setattr__(self, _BACKING_ATTR, value)

    transformers.PreTrainedModel.all_tied_weights_keys = property(_getter, _setter)
    transformers.PreTrainedModel._benchmark_tied_weights_patched = True


class InternVLModel(BaseModel):

    name = "InternVLModel"

    def load(self) -> None:
        checkpoint = self.checkpoint_path or DEFAULT_CHECKPOINTS.get(self.name)
        if not checkpoint:
            raise ValueError(f"{self.name}: pass --checkpoint {self.name}=<hf repo id or local path>")
        import torch
        from transformers import AutoModel, AutoTokenizer

        _patch_tied_weights_compat()

        self.torch = torch
        world_size = torch.cuda.device_count()
        # Vision tower always lands on GPU 0 in the split map below (or the
        # single --device when there's only one GPU) -- pixel_values need to
        # be sent there in predict().
        self._input_device = 0 if world_size > 1 else self.device

        load_kwargs = dict(
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if world_size > 1:
            # needs sharding across multiple GPUs
            load_kwargs["device_map"] = split_model_across_gpus(checkpoint)
        
        try:
            self.model = AutoModel.from_pretrained(checkpoint, use_flash_attn=True, **load_kwargs).eval()
        except (ImportError, ValueError) as e:
            logging.warning(f"{self.name}: use_flash_attn=True failed ({e}); retrying without flash-attn")
            self.model = AutoModel.from_pretrained(checkpoint, use_flash_attn=False, **load_kwargs).eval()

        if world_size <= 1:
            self.model = self.model.to(self.device)

        _assert_fully_loaded(self.model, self.name)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
        self._loaded = True

    def unload(self) -> None:
        if self._loaded:
            del self.model
        
            import gc

            gc.collect()
            self.torch.cuda.empty_cache()
        self._loaded = False

    def predict(self, image_path: str) -> str:
        import torch

        pixel_values_list = _internvl_load_image(image_path, max_num=12)
        pixel_values = torch.stack(pixel_values_list).to(self._input_device, dtype=torch.bfloat16)
        # repetition_penalty=1.0 -- same reasoning as the Qwen classes:
        # avoids RepetitionPenaltyLogitsProcessor gathering scores using
        # image-placeholder token ids that may fall outside the LM head's
        # output range, and is meaningless for an 8-token classification
        # task regardless.
        generation_config = dict(max_new_tokens=8, do_sample=False, repetition_penalty=1.0)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, INTERNVL_PROMPT, generation_config)
        return parse_real_fake_tampered(response)


# ---------------------------------------------------------------------------
# Registry: name -> class. 
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, type] = {
    "Qwen-2.5VL-7B": QwenVLModel,
    "InternVL3-8B": InternVLModel,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    model_name: str
    # subset -> {css_level: accuracy or None}
    accuracy: Dict[str, Dict[int, Optional[float]]] = field(default_factory=dict)
    # subset -> caAcc or None, one dict per weighting curve
    ca_acc_linear: Dict[str, Optional[float]] = field(default_factory=dict)
    ca_acc_fib: Dict[str, Optional[float]] = field(default_factory=dict)
    n_errors: int = 0
    error_message: Optional[str] = None


def _cleanup_after_failed_load(model: BaseModel) -> None:
    """Called when model.load() raises partway through -- e.g.
    _assert_fully_loaded() rejecting a too-small --gpu selection AFTER
    from_pretrained() already allocated real GPU memory for self.model.
    """
    try:
        import torch

        if hasattr(model, "model"):
            del model.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as cleanup_err:
        logging.warning(f"{model.name}: cleanup after failed load() also failed: {cleanup_err}")


def evaluate_model(
    model: BaseModel, samples: List[Sample], verbose: bool = True
) -> ModelResult:
    correct = {s: {c: 0 for c in CSS_LEVELS} for s in SUBSETS}
    total = {s: {c: 0 for c in CSS_LEVELS} for s in SUBSETS}
    n_errors = 0

    try:
        model.load()
    except NotImplementedError as e:
        logging.warning(f"{model.name}: {e}")
        return ModelResult(model_name=model.name, error_message=str(e))
    except Exception as e:
        # exc_info=True -- str(e) alone was hiding exactly where inside
        # load() this broke (e.g. a bad attribute access deep in a
        # trust_remote_code model's from_pretrained()/tie_weights() path).
        # The full traceback is what actually pinpoints the faulting call.
        logging.warning(f"{model.name}: load() failed: {e}", exc_info=True)
        _cleanup_after_failed_load(model)
        return ModelResult(model_name=model.name, error_message=f"load() failed: {e}")

    t0 = time.time()
    for i, sample in enumerate(samples):
        total[sample.label][sample.css_level] += 1
        # Every registered model is a VLM prompted to answer real/fake/tampered,
        # so it's always scored against the sample's exact label.
        expected = sample.label
        try:
            pred = model.predict(sample.image_path)
            if pred == expected:
                correct[sample.label][sample.css_level] += 1
        except Exception as e:
            n_errors += 1
            if verbose and n_errors <= 3:
                # exc_info=True logs the full traceback, not just str(e) --
                # with --cuda-launch-blocking on, that traceback points at
                # the actual faulting op instead of a generic CUDA assert
                # message. 
                logging.warning(f"{model.name}: predict failed on {sample.image_path}: {e}", exc_info=True)
        if verbose and (i + 1) % 10 == 0:
            logging.info(f"{model.name}: {i + 1}/{len(samples)} images...")

    try:
        model.unload()
    except Exception as e:
        logging.warning(f"{model.name}: unload() failed, keeping computed results anyway: {e}", exc_info=True)
    elapsed = time.time() - t0
    if verbose:
        logging.info(f"{model.name}: done in {elapsed:.1f}s ({n_errors} prediction errors)")

    accuracy: Dict[str, Dict[int, Optional[float]]] = {}
    ca_acc_linear: Dict[str, Optional[float]] = {}
    ca_acc_fib: Dict[str, Optional[float]] = {}
    for s in SUBSETS:
        css_acc: Dict[int, Optional[float]] = {}
        for c in CSS_LEVELS:
            css_acc[c] = (100.0 * correct[s][c] / total[s][c]) if total[s][c] > 0 else None
        accuracy[s] = css_acc
        ca_acc_linear[s] = compute_ca_acc(css_acc, CSS_WEIGHTS_LINEAR)
        ca_acc_fib[s] = compute_ca_acc(css_acc, CSS_WEIGHTS_FIBONACCI)

    return ModelResult(
        model_name=model.name,
        accuracy=accuracy,
        ca_acc_linear=ca_acc_linear,
        ca_acc_fib=ca_acc_fib,
        n_errors=n_errors,
    )


def compute_ca_acc(css_accuracy: Dict[int, Optional[float]], css_weights: Dict[int, float]) -> Optional[float]:
    """Content Aware Accuracy: weighted sum of per-CSS-level accuracy.

    Weights are renormalized over whichever CSS levels actually have data,
    so a subset missing e.g. CSS level 4 still produces a valid score.
    """
    present = {c: a for c, a in css_accuracy.items() if a is not None}
    if not present:
        return None
    weight_sum = sum(css_weights[c] for c in present)
    if weight_sum == 0:
        return None
    return sum(css_weights[c] * a for c, a in present.items()) / weight_sum


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_cell(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def build_rows(results: List[ModelResult], subset: str):
    headers = ["Model", "CSS_1", "CSS_2", "CSS_3", "CSS_4", "caAcc_linear", "caAcc_fib"]
    rows = []
    for r in results:
        if r.error_message:
            rows.append([r.model_name, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"])
        else:
            css_acc = r.accuracy.get(subset, {})
            rows.append(
                [
                    r.model_name,
                    fmt_cell(css_acc.get(1)),
                    fmt_cell(css_acc.get(2)),
                    fmt_cell(css_acc.get(3)),
                    fmt_cell(css_acc.get(4)),
                    fmt_cell(r.ca_acc_linear.get(subset)),
                    fmt_cell(r.ca_acc_fib.get(subset)),
                ]
            )
    return headers, rows


def print_subset_table(results: List[ModelResult], subset: str) -> None:
    # Builds the whole table as one string and logs it as a single entry
    lines = [SUBSET_TITLES[subset], "-" * len(SUBSET_TITLES[subset])]
    headers, rows = build_rows(results, subset)
    widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(len(headers))]

    def fmt_row(row):
        return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines.append(fmt_row(headers))
    lines.append(sep)
    for row in rows:
        lines.append(fmt_row(row))

    errors = [r for r in results if r.error_message]
    if errors:
        lines.append("")
        lines.append("Models skipped (not yet wired up / failed to load):")
        for r in errors:
            lines.append(f"  - {r.model_name}: {r.error_message}")

    logging.info("\n" + "\n".join(lines))


def write_subset_csv(results: List[ModelResult], subset: str, output_csv: str) -> None:
    headers, rows = build_rows(results, subset)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def csv_path_for_subset(output_csv: str, subset: str) -> str:
    p = Path(output_csv)
    return str(p.with_name(f"{p.stem}_{subset}{p.suffix or '.csv'}"))


def parse_checkpoint_args(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--checkpoint must be NAME=PATH, got {p!r}")
        name, path = p.split("=", 1)
        out[name] = path
    return out


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLMs on a CSS-labeled dataset.")
    parser.add_argument("--manifest", default="/mnt/ssd1/bairat/metadata/dataset2/manifest_css.json", help="JSON manifest (manifest_css.py format) or generic CSV")
    parser.add_argument("--images-dir", default="/mnt/ssd1/bairat/dataset/dataset2", help="Base dir images are searched under (several layouts auto-tried)")
    parser.add_argument("--path-template", default=None, help="Custom template, e.g. '{source_dataset}/{label}/{filename}'")
    parser.add_argument("--col-image", default="image_path", help="(generic CSV fallback only)")
    parser.add_argument("--col-label", default="label", help="(generic CSV fallback only)")
    parser.add_argument("--col-css", default="css_level", help="(generic CSV fallback only)")
    parser.add_argument(
        "--models",
        default="all",
        help=f"Comma-separated model names to run, or 'all'. Available: {', '.join(MODEL_REGISTRY)}",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "NAME=<hf repo id or local path>, repeatable. Optional for "
            f"{', '.join(DEFAULT_CHECKPOINTS)} (they default to a Hub id -- see DEFAULT_CHECKPOINTS); "
            "pass this to override with a different checkpoint/revision."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu",
        type=str,
        default="3",
        help=(
            "GPU id(s) to restrict to, e.g. '2' or '0,1' -- sets CUDA_VISIBLE_DEVICES "
            "before any model loads, same as the --gpu flag in your CSS-scoring scripts. "
            "Defaults to '3'; pass --gpu '' to leave your shell's CUDA_VISIBLE_DEVICES untouched."
        ),
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default="/mnt/ssd1/bairat/models"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help=(
            "Hugging Face Hub access token, sets HF_TOKEN before any model loads."
        ),
    )
    parser.add_argument(
        "--disable-xet",
        type=str,
        default="1",
        help=(
            "Sets HF_HUB_DISABLE_XET before any model loads, falling back to plain HTTP "
            "downloads instead of Hugging Face Hub's Xet backend."
        ),
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help=(
            "Sets CUDA_LAUNCH_BLOCKING=1 before any model loads, making CUDA calls synchronous "
            "so a 'device-side assert triggered' error points at the actual faulting operation "
            "instead of a generic CUDA assert message. This is useful for debugging "
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="/mnt/ssd1/bairat/results/benchmark_results.csv",
        help="Base path; writes _real/_fake/_tampered CSVs here (directory is created if needed)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help=(
            "Log file path (default: just the basename of --output-csv, with a .log suffix "
        ),
    )
    args = parser.parse_args()

    # Logging setup: everything from here on is written to a log file,
    # not the console
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Log defaults to the CURRENT directory (just output_path's basename,
    # not its full path) -- deliberately independent of --output-csv's
    # directory, so results can be written to a shared/remote results drive
    # while the log stays local. Pass --log explicitly to put it anywhere else.
    log_path = Path(args.log) if args.log else Path(output_path.name).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1)

    handler = logging.StreamHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    console = os.fdopen(os.dup(sys.stderr.fileno()), "w")

    print(f"Progress is logged to {log_path}, not this console -- tail -f it to watch live", file=console)
    logging.info(f"=== {Path(sys.argv[0]).name} starting -- output CSVs will be at {args.output_csv} (subset-suffixed) ===")


    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())

    if args.hf_home:
        Path(args.hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = args.hf_home
        logging.info(f"HF cache directory: {args.hf_home}")

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        logging.info("HF_TOKEN set from --hf-token/$HF_TOKEN")
    else:
        logging.warning(
            "no --hf-token / $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN set -- gated repos will "
            "fail to download and anonymous requests may get rate-limited. Falling back to any "
            "cached `huggingface-cli login` credentials, if present."
        )

    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = args.disable_xet
        logging.info("Xet downloads disabled (HF_HUB_DISABLE_XET=1) -- using plain HTTP")

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logging.info(f"Using GPU(s): {args.gpu}")

    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logging.info("CUDA_LAUNCH_BLOCKING=1 -- CUDA calls are synchronous (slower, for debugging a crash)")

    model_names = list(MODEL_REGISTRY.keys()) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    unknown = [m for m in model_names if m not in MODEL_REGISTRY]
    if unknown:
        parser.error(f"Unknown model(s): {unknown}. Available: {list(MODEL_REGISTRY)}")

    checkpoints = parse_checkpoint_args(args.checkpoint)

    logging.info(f"Loading manifest: {args.manifest}")
    samples = load_manifest(
        args.manifest, args.images_dir, args.path_template, args.col_image, args.col_label, args.col_css
    )
    logging.info(f"Loaded {len(samples)} samples.")
    for s in SUBSETS:
        subset_samples = [x for x in samples if x.label == s]
        logging.info(f"  {s}: {len(subset_samples)} images")
        for c in CSS_LEVELS:
            n = sum(1 for x in subset_samples if x.css_level == c)
            logging.info(f"    CSS level {c}: {n} images")

    if args.images_dir:
        missing = sum(1 for s in samples if not Path(s.image_path).exists())
        if missing:
            logging.warning(
                f"{missing}/{len(samples)} image files were not found under --images-dir "
                f"with any of the auto-tried layouts. Consider --path-template. "
                f"Example unresolved path: {next(s.image_path for s in samples if not Path(s.image_path).exists())}"
            )

    results: List[ModelResult] = []
    for name in model_names:
        cls = MODEL_REGISTRY[name]
        logging.info(f"=== {name} ===")
        model = cls(checkpoint_path=checkpoints.get(name), device=args.device, model_name=name)
        try:
            result = evaluate_model(model, samples)
        except Exception as e:
            logging.warning(f"{name}: evaluate_model() crashed unexpectedly: {e}", exc_info=True)
            result = ModelResult(model_name=name, error_message=f"crashed during evaluation: {e}")
        results.append(result)

        # Write out after every model (not just at the end) so a later
        # model crashing never loses results already computed for earlier
        # ones -- each subset CSV always reflects everything finished so far.
        for s in SUBSETS:
            write_subset_csv(results, s, csv_path_for_subset(args.output_csv, s))

    logging.info("=" * 70)
    logging.info("RESULTS")
    logging.info("=" * 70)
    for s in SUBSETS:
        print_subset_table(results, s)
        out_path = csv_path_for_subset(args.output_csv, s)
        logging.info(f"Saved CSV to: {out_path}")

    print(f"Done -- results written to {args.output_csv} (subset-suffixed). Full log: {log_path}", file=console)


if __name__ == "__main__":
    main()