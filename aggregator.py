"""
Aggregator — Qwen2.5-7B-Instruct (text-only)
Reads the 3 proposer output files, synthesizes a final CSS score per image.
Writes results back into the manifest as manifest_css.json.


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
# PROMPTS
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior editorial director responsible for final content sensitivity decisions.
You receive assessments from multiple expert analysts and must synthesize them into a single authoritative score."""

PROMPT_TEMPLATE = """You have received Content Sensitivity Score (CSS) assessments from 3 expert analysts for the same image.

CSS Scale:
  1 = Everyday, non-sensitive content
  2 = Mildly sensitive
  3 = Moderately sensitive
  4 = Highly sensitive

Analyst Assessments:
{assessments}

Instructions:
1. Carefully review each analyst's score and justification.
2. Consider points of agreement and disagreement.
3. Weigh the quality of each justification, not just the score.
4. Assign a final CSS score that best reflects the image's true sensitivity.

You MUST respond in this exact JSON format and nothing else:
{{
  "consensus_level": "<high|medium|low>",
  "agreement_analysis": "<brief analysis of where analysts agreed or disagreed>",
  "final_css_score": <integer 1-4>,
  "final_justification": "<2-3 sentence explanation of the final score>"
}}"""

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

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


def load_proposer_results(paths):
    """
    Load the 3 proposer output files.
    Returns a dict: image_id -> {model_name: parsed_result}
    """
    per_image = {}  # image_id -> {model_name: parsed or None}

    for model_name, path in paths.items():
        path = Path(path)
        if not path.exists():
            logging.warning(f"proposer output not found: {path}")
            continue
        with open(path) as f:
            results = json.load(f)
        for entry in results:
            image_id = entry["image_id"]
            if image_id not in per_image:
                per_image[image_id] = {}
            per_image[image_id][model_name] = entry.get("parsed")

    return per_image


def format_assessments(proposer_outputs):
    """Format proposer outputs into a readable string for the aggregator prompt."""
    text = ""
    for i, (model_name, result) in enumerate(proposer_outputs.items(), 1):
        if result is None:
            text += f"\nAnalyst {i} ({model_name}): [NO RESPONSE]\n"
        else:
            text += f"""
Analyst {i} ({model_name}):
  CSS Score: {result.get("css_score", "N/A")}
  Scene: {result.get("scene_description", "N/A")}
  Key factors: {", ".join(result.get("key_sensitivity_factors", []))}
  Justification: {result.get("justification", "N/A")}
"""
    return text


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.info(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def aggregate(proposer_outputs, model, tokenizer):
    """Run the aggregator on one image's proposer outputs."""
    assessments = format_assessments(proposer_outputs)
    prompt = PROMPT_TEMPLATE.format(assessments=assessments)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(trimmed[0], skip_special_tokens=True)


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
    parser = argparse.ArgumentParser(description="CSS Aggregator")
    parser.add_argument("--manifest",         type=str, default="/mnt/ssd1/bairat/metadata/dataset3/manifest.json")
    parser.add_argument("--pixtral_scores",  type=str, default="/mnt/ssd1/bairat/css_scores/pixtral.json")
    parser.add_argument("--qwen_scores",      type=str, default="/mnt/ssd1/bairat/css_scores/qwen.json")
    parser.add_argument("--internvl_scores",  type=str, default="/mnt/ssd1/bairat/css_scores/internvl.json")
    parser.add_argument("--output",           type=str, default="/mnt/ssd1/bairat/metadata/dataset3/manifest_css.json")
    parser.add_argument("--log",              type=str, default=None,
                        help="Log file path (default: <output> with a .log suffix). All "
                             "status/warnings/errors and the tqdm progress bar are written "
                             "here instead of the console, so the process can run detached "
                             "in tmux without depending on a terminal being attached.")
    parser.add_argument("--gpu",              type=str, default="2")
    parser.add_argument("--resume",           action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Load manifest
    with open(args.manifest) as f:
        entries = json.load(f)
    logging.info(f"Loaded {len(entries)} manifest entries")

    # Load all proposer results
    proposer_paths = {
        "pixtral-12B":   args.pixtral_scores,
        "qwen2.5-vl":   args.qwen_scores,
        "internvl3":    args.internvl_scores
    }
    per_image = load_proposer_results(proposer_paths)
    logging.info(f"Proposer results loaded for {len(per_image)} images")

    # Load existing output if resuming
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = {e["image_id"]: e for e in json.load(f)}
        logging.info(f"Resuming — {sum(1 for e in existing.values() if e.get('final_css_score') is not None)} already aggregated")
    else:
        existing = {}

    model, tokenizer = load_model()

    results = []
    for entry in tqdm(entries, desc="Aggregating", file=log_file, mininterval=5.0):
        image_id = entry["image_id"]

        # Resume: skip if already aggregated
        if args.resume and image_id in existing:
            if existing[image_id].get("final_css_score") is not None:
                results.append(existing[image_id])
                continue

        # Get proposer outputs for this image
        proposer_outputs = per_image.get(image_id, {})
        n_available = sum(1 for v in proposer_outputs.values() if v is not None)

        if n_available == 0:
            logging.info(f"{image_id} → no proposer results available, skipping")
            entry["css_proposer_responses"]  = proposer_outputs
            entry["css_aggregated_response"] = None
            entry["final_css_score"]         = None
            results.append(entry)
            continue

        try:
            raw    = aggregate(proposer_outputs, model, tokenizer)
            parsed = parse_json_response(raw)
            if parsed:
                logging.info(f"{image_id} → final CSS {parsed.get('final_css_score')} "
                             f"(consensus: {parsed.get('consensus_level')})")
            else:
                logging.warning(f"{image_id} → parse failed | raw: {raw[:100]}")
        except Exception as e:
            logging.error(f"{image_id} → ERROR: {e}")
            raw    = None
            parsed = None

        entry["css_proposer_responses"]  = proposer_outputs
        entry["css_aggregated_response"] = parsed
        entry["final_css_score"]         = parsed.get("final_css_score") if parsed else None
        results.append(entry)

        # Save after every image
        atomic_write_json(output_path, results)

    logging.info(f"DONE — {len(results)} images aggregated. Output: {output_path}")
    print(f"Done — {len(results)} images aggregated. Output: {output_path}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()