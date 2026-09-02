"""
Proposer 1 — Pixtral-12B-2409
Scores all images in the manifest with a CSS score (1-4).
Writes results to /mnt/ssd1/bairat/css_scores/pixtral.json

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
import base64
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

SYSTEM_PROMPT = """You are an expert media analyst specializing in content sensitivity assessment.
Your role is to analyze images and assign a Content Sensitivity Score (CSS) based on their visual content."""

USER_PROMPT = """Analyze this image and assign a Content Sensitivity Score (CSS).

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

MODEL_PATH = "/mnt/ssd1/bairat/models/pixtral/PIXTRAL-12B-2409"


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


def image_to_base64(image_path):
    suffix = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def load_model():
    from mistral_inference.transformer import Transformer
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    logging.info(f"Loading Pixtral from {MODEL_PATH}...")
    tokenizer = MistralTokenizer.from_file(f"{MODEL_PATH}/tekken.json")
    model = Transformer.from_folder(MODEL_PATH, dtype=torch.bfloat16, device="cuda")
    model = model.to("cuda").eval()
    return model, tokenizer


def score_image(image_path, model, tokenizer):
    from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageChunk
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_inference.generate import generate

    image_data = image_to_base64(image_path)
    # strip the data URL prefix, keep only base64 + media type
    header, b64 = image_data.split(",", 1)
    media_type = header.split(":")[1].split(";")[0]

    from PIL import Image as PILImage

    request = ChatCompletionRequest(
        messages=[
            UserMessage(content=[
                ImageChunk(image=PILImage.open(image_path).convert("RGB")),
                TextChunk(text=SYSTEM_PROMPT + "\n\n" + USER_PROMPT),
            ])
        ],
        model="pixtral",
    )

    tokenized = tokenizer.encode_chat_completion(request)
    input_ids = tokenized.tokens
    images = tokenized.images

    out_tokens, _ = generate(
        [input_ids],
        model,
        images=[images],
        max_tokens=512,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )

    raw = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return raw


def load_existing_results(output_path):
    """
    Load prior results keyed by image_id (a dict), so that
    retrying a previously-failed image overwrites its old record.

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


def main():
    parser = argparse.ArgumentParser(description="Pixtral-12B CSS Proposer")
    parser.add_argument("--manifest",   type=str, default="/mnt/ssd1/bairat/metadata/dataset3/manifest.json")
    parser.add_argument("--images_dir", type=str, default="/mnt/ssd1/bairat/dataset/dataset3")
    parser.add_argument("--output",     type=str, default="/mnt/ssd1/bairat/css_scores/pixtral.json")
    parser.add_argument("--log",        type=str, default=None,
                            help="Log file path (default: <output> with a .log suffix). All "
                                "status/warnings/errors and the tqdm progress bar are written "
                                "here instead of the console, so the process can run detached "
                                "in tmux without depending on a terminal being attached.")
    parser.add_argument("--gpu",        type=str, default="0")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip images already successfully scored in --output; "
                             "automatically retries images that previously failed or were missing.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)

    # ── Logging setup: one shared, line-buffered file handle used for both
    # the logging module and tqdm's progress bar, so nothing from this
    # script writes to stdout/stderr once it's past this point. ──
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

    for entry in tqdm(entries, desc="Pixtral-12B", file=log_file, mininterval=5.0):
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
        # and the comment above) -- no need to rescan all of `entries` here
        # like the original did. Written atomically so a kill mid-write
        # can't corrupt the file.
        atomic_write_json(output_path, list(results.values()))

    logging.info(f"Done — {len(results)} images scored. Output: {output_path}")
    print(f"Done — {len(results)} images scored. Output: {output_path}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()