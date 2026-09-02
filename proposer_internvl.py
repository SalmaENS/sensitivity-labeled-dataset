"""
Proposer 3 — InternVL3-8B
Scores all images in the manifest with a CSS score (1-4).
Writes results to /mnt/ssd1/bairat/css_scores/internvl.json

Use --resume to only score images that aren't already successfully scored
(missing/failed images from a previous run are automatically retried).

This can run detached in tmux without depending on the console
1. All status/warnings/errors,
   and the tqdm progress bar itself, are written to a log file (--log,
   default <output>.log) instead of the console.
2. The output JSON is written atomically: serialize to a temp file,
   then os.replace() it over the real path, so a kill/crash mid-write
   can't truncate/corrupt manifest_css.json.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────────────────────

USER_PROMPT = """<image>
Analyze this image and assign a Content Sensitivity Score (CSS).

CSS Scale:
  1 = Everyday, non-sensitive content (landscapes, food, casual scenes, everyday life)
  2 = Mildly sensitive (public figures in neutral contexts, minor accidents, protests)
  3 = Moderately sensitive (political events, medical content, social conflicts)
  4 = Highly sensitive (war imagery, severe accidents, crisis situations, graphic violence)

Instructions:
1. Carefully examine the scene, people, objects, and context in the image.
2. Identify the key elements that determine its sensitivity level.
3. Assign a single integer CSS score from 1 to 4.

You MUST respond in this exact JSON format and nothing else:
{
  "scene_description": "<brief factual description of what is shown>",
  "key_sensitivity_factors": ["<factor1>", "<factor2>"],
  "css_score": <integer 1-4>,
  "justification": "<2-3 sentence explanation of why this score was assigned>"
}"""

MODEL_NAME = "OpenGVLab/InternVL3-8B"

# ──────────────────────────────────────────────────────────────
# INTERNVL3 IMAGE PREPROCESSING
# ──────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width  = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width  // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width  // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image_internvl(pil_image, input_size=448, max_num=12):
    transform = build_transform(input_size)
    images = dynamic_preprocess(pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])

def split_model_across_gpus(model_name):
    from transformers import AutoConfig
    device_map  = {}
    world_size  = torch.cuda.device_count()
    config      = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers  = config.llm_config.num_hidden_layers
    num_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_per_gpu = [num_per_gpu] * world_size
    num_per_gpu[0] = math.ceil(num_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, n in enumerate(num_per_gpu):
        for _ in range(n):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    for key in ["vision_model", "mlp1", "language_model.model.tok_embeddings",
                "language_model.model.embed_tokens", "language_model.output",
                "language_model.model.norm", "language_model.model.rotary_emb",
                "language_model.lm_head"]:
        device_map[key] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def parse_json_response(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def load_model():
    from transformers import AutoModel, AutoTokenizer

    logging.info(f"Loading {MODEL_NAME}...")
    device_map = split_model_across_gpus(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


def score_image(image_path, model, tokenizer):
    pil_image    = Image.open(image_path).convert("RGB")
    pixel_values = load_image_internvl(pil_image, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=512, do_sample=False)
    if tokenizer.pad_token_id is None:
        generation_config["pad_token_id"] = tokenizer.eos_token_id
    return model.chat(tokenizer, pixel_values, USER_PROMPT, generation_config)


def load_existing_results(output_path):
    """
    Load prior results keyed by image_id (a dict, not a list), so that
    retrying a previously-failed image overwrites its old record instead of
    appending a duplicate entry for the same image_id.

    """
    with open(output_path) as f:
        raw_results = json.load(f)
    results = {r["image_id"]: r for r in raw_results}
    done_ids = {image_id for image_id, r in results.items() if r.get("raw_response") is not None}
    return results, done_ids


def atomic_write_json(path: Path, data) -> None:
    """
    Write `data` as JSON to `path` atomically: serialize to a temp file in
    the same directory, then os.replace() it over the real path, so a
    crash or kill mid-write can never leave `path` truncated or corrupted.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InternVL3 CSS Proposer")
    parser.add_argument("--manifest",   type=str, default="/mnt/ssd1/bairat/metadata/dataset3/manifest.json")
    parser.add_argument("--images_dir", type=str, default="/mnt/ssd1/bairat/dataset/dataset3")
    parser.add_argument("--output",     type=str, default="/mnt/ssd1/bairat/css_scores/internvl.json")
    parser.add_argument("--log",        type=str, default=None,
                        help="Log file path (default: <output> with a .log suffix). All "
                            "status/warnings/errors and the tqdm progress bar are written "
                            "here instead of the console, so the process can run detached "
                            "in tmux without depending on a terminal being attached.")
    parser.add_argument("--gpu",        type=str, default="3")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip images already successfully scored in --output; "
                             "automatically retries images that previously failed or were missing.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)

    # Logging setup: one shared, line-buffered file handle used for both
    # the logging module and tqdm's progress bar
    log_path = Path(args.log) if args.log else output_path.with_suffix(".log")
    log_file = open(log_path, "a", buffering=1)

    handler = logging.StreamHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    print(f"Using GPU(s): {args.gpu} -- progress is logged to {log_path}, not this console", file=sys.stderr)
    logging.info(f"Using GPU(s): {args.gpu}")

    with open(args.manifest) as f:
        entries = json.load(f)
    logging.info(f"Loaded {len(entries)} entries")

    if args.resume and output_path.exists():
        results, done_ids = load_existing_results(output_path)
        to_retry = len(results) - len(done_ids)
        logging.info(f"Resuming — {len(done_ids)} already scored, {to_retry} previously failed/missing will be retried")
    else:
        results = {}
        done_ids = set()

    model, tokenizer = load_model()

    for entry in tqdm(entries, desc="InternVL3", file=log_file, mininterval=5.0):
        image_id   = entry["image_id"]
        image_path = images_dir / entry["filename"]

        if image_id in done_ids:
            continue

        if not image_path.exists():
            logging.warning(f"{image_path} not found")
            results[image_id] = {"image_id": image_id, "raw_response": None, "parsed": None}
            continue

        try:
            raw    = score_image(image_path, model, tokenizer)
            parsed = parse_json_response(raw)
            if parsed:
                logging.info(f"{image_id} → CSS {parsed.get('css_score')}")
            else:
                logging.warning(f"{image_id} → parse failed | raw: {raw[:100]}")
        except Exception as e:
            logging.error(f"{image_id} → ERROR: {e}")
            raw    = None
            parsed = None

        results[image_id] = {"image_id": image_id, "raw_response": raw, "parsed": parsed}

        # `results` is already in manifest order (see load_existing_results
        # and the comment above). Written atomically so a kill mid-write
        # can't corrupt the file.
        atomic_write_json(output_path, list(results.values()))

    logging.info(f"Done — {len(results)} images scored. Output: {output_path}")
    print(f"Done — {len(results)} images scored. Output: {output_path}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()