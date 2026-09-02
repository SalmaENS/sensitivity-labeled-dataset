"""
Proposer 2 — Qwen2.5-VL-7B-Instruct
Scores all images in the manifest with a CSS score (1-4).
Writes results to /mnt/ssd1/bairat/css_scores/qwen.json

Use --resume to only score images that aren't already successfully scored
(missing/failed/unparsed images from a previous run are automatically retried).


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
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────────────────────

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

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

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
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    logging.info(f"Loading {MODEL_NAME}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def score_image(image_path, model, processor):
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text",  "text": USER_PROMPT},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def load_existing_results(output_path):
    """
    Load prior results keyed by image_id (a dict, not a list), so that
    retrying a previously-failed image overwrites its old record instead of
    appending a duplicate entry for the same image_id.


    """
    with open(output_path) as f:
        raw_results = json.load(f)
    results = {r["image_id"]: r for r in raw_results}
    done_ids = {image_id for image_id, r in results.items() if r.get("parsed") is not None}
    return results, done_ids


def atomic_write_json(path: Path, data) -> None:
    """
    Write `data` as JSON to `path` atomically: serialize to a temp file in
    the same directory, then os.replace() it over the real path. os.replace
    is atomic on the same filesystem, so a crash or kill mid-write can never
    leave `path` truncated or corrupted.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL CSS Proposer")
    parser.add_argument("--manifest",   type=str, default="/mnt/ssd1/bairat/metadata/dataset3/manifest.json")
    parser.add_argument("--images_dir", type=str, default="/mnt/ssd1/bairat/dataset/dataset3")
    parser.add_argument("--output",     type=str, default="/mnt/ssd1/bairat/css_scores/qwen.json")
    parser.add_argument("--log",        type=str, default=None,
                                help="Log file path (default: <output> with a .log suffix). All "
                                    "status/warnings/errors and the tqdm progress bar are written "
                                    "here instead of the console, so the process can run detached "
                                    "in tmux without depending on a terminal being attached.")
    parser.add_argument("--gpu",        type=str, default="2")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip images already successfully parsed in --output; "
                             "automatically retries images that previously failed, were "
                             "missing, or produced an unparseable response.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)

    # ── Logging setup: one shared, line-buffered file handle used for both
    # the logging module and tqdm's progress bar, so absolutely nothing from
    # this script writes to stdout/stderr once it's past this point. ──
    log_path = Path(args.log) if args.log else output_path.with_suffix(".log")
    log_file = open(log_path, "a", buffering=1)

    handler = logging.StreamHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    # One line to the real console so a human launching this interactively
    # knows where the actual log lives -- this is the only thing ever
    # written to stdout/stderr, so it can't fill up a stalled pty.
    print(f"Using GPU(s): {args.gpu} -- progress is logged to {log_path}, not this console", file=sys.stderr)
    logging.info(f"Using GPU(s): {args.gpu}")

    with open(args.manifest) as f:
        entries = json.load(f)
    logging.info(f"Loaded {len(entries)} entries")

    if args.resume and output_path.exists():
        results, done_ids = load_existing_results(output_path)
        to_retry = len(results) - len(done_ids)
        logging.info(f"Resuming — {len(done_ids)} already scored, {to_retry} previously failed/missing/unparsed will be retried")
    else:
        results = {}
        done_ids = set()

    model, processor = load_model()

    for entry in tqdm(entries, desc="Qwen2.5-VL", file=log_file, mininterval=5.0):
        image_id   = entry["image_id"]
        image_path = images_dir / entry["filename"]

        if image_id in done_ids:
            continue

        if not image_path.exists():
            logging.warning(f"{image_path} not found")
            results[image_id] = {"image_id": image_id, "raw_response": None, "parsed": None}
            continue

        try:
            raw    = score_image(image_path, model, processor)
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