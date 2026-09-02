"""
Phase 1 - Sample images for the dataset.

There's no overall --dataset_size : every
(dataset, label) pair has its own explicit count argument, and that's
exactly how many images get sampled for it. The arguments are:
  - RRDataset:       --n_rrdataset_real, --n_rrdataset_fake (the fake count
                      is spread evenly across the 5 fake scenarios — there's
                      no per-scenario argument)
  - DGM4:            --n_dgm4_real, --n_dgm4_tampered
  - SID-Set:         --n_sidset_real, --n_sidset_fake, --n_sidset_tampered
  - Sens-VisualNews: --n_sensvisualnews_real (internally split evenly across
                      4 keyword buckets for diversity — see
                      sample_sens_visualnews)

At startup, the script prints each dataset's total (sum of its label
counts), its label breakdown, and what percentage of the combined total
(the sum across all datasets) it represents.

The manifest is written incrementally (after every image, not just at the
end of the run) via ManifestWriter, using an atomic write-then-rename so a
crash or kill mid-run can't corrupt the file or silently lose metadata for
images that already made it to disk.

Source datasets:
  - RRDataset       (real, fake)             -- alecrespi/RRDataset-CV-Project2
  - DGM4            (real, tampered)         -- rshaojimmy/DGM4
  - SID-Set         (real, fake, tampered)   -- saberzl/SID_Set
  - Sens-VisualNews (real only)              -- read directly out of a local
                                                 VisualNews origin.tar (no
                                                 public HF repo)
"""

import argparse
import json
import os
import random
import re
import sys
import io
import tarfile
from datetime import date
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def save_image(pil_image, path):
    """Save a PIL image as JPEG, converting mode if needed."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(path, format="JPEG", quality=95)


def make_entry(image_id, source, label, sensitivity, scenario,
               generator=None, original_label=None, hf_repo=None, seed=42):
    """Build a metadata dict with all Phase 2 fields pre-populated as null."""
    return {
        "image_id": image_id,
        "filename": f"{image_id}.jpg",
        "source_dataset": source,
        "hf_repo": hf_repo,
        "label": label,                        # "real" or "fake" or "tampered"
        "sensitivity_tier": sensitivity,
        "scenario": scenario,
        "generator_model": generator,
        "original_label_value": original_label,
        "css_score": None,
        "css_proposer_responses": [],
        "css_aggregated_response": None,
        "sampling_date": str(date.today()),
        "random_seed": seed,
    }


def distribute_evenly(total, keys):
    """
    Split `total` as evenly as possible across `keys`, returning {key: count}.
    Counts differ by at most 1; any remainder goes to the first keys in order,
    so results are deterministic given the same `keys` ordering.
    """
    n = len(keys)
    base, remainder = divmod(total, n)
    return {key: base + (1 if idx < remainder else 0) for idx, key in enumerate(keys)}


class ManifestWriter:
    """
    Holds the full list of manifest entries in memory and flushes to disk
    after every addition, so metadata is never more than one image behind
    what's actually saved to disk — even if the process is killed mid-run.

    Writes are atomic: each flush writes to a temp file in the same
    directory, then renames it over the real path (rename is atomic on the
    same filesystem), so a crash mid-write can't leave a half-written or
    corrupted manifest.json behind.
    """

    def __init__(self, metadata_path):
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.entries = json.load(f)
        else:
            self.entries = []

    def add(self, entry):
        self.entries.append(entry)
        self._flush()

    def _flush(self):
        tmp_path = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.entries, f, indent=2)
        os.replace(tmp_path, self.metadata_path)  

    def __len__(self):
        return len(self.entries)


# ──────────────────────────────────────────────────────────────
# DATASET 1 — RRDataset
# Columns: jpg (image bytes), __key__ (encodes label + scenario)
# Key format: "RRDataset_.../split/ai|real/Scenario_Name_XXXXXX"
# ──────────────────────────────────────────────────────────────

# The 5 "fake" (ai) scenarios in RRDataset that we sample from, plus the
# single "real" bucket. scenario_targets passed to sample_rrdataset() uses
# these exact keys.
RR_FAKE_SCENARIOS = [
    "War_&_Conflict_Scenes",
    "Natural_Disasters_&_Accidents",
    "Political_&_Social_Events",
    "Medical_&_Public_Health",
    "Culture_&_Religion",
]
RR_REAL_BUCKET = "real"


def sample_rrdataset(output_dir, scenario_targets, seed, hf_token, manifest):
    """
    scenario_targets: dict mapping bucket -> count, where bucket is either
    "real" (label=real) or one of RR_FAKE_SCENARIOS (label=fake). Buckets
    with a target of 0 (or absent) are skipped entirely.
    """
    print("\n" + "="*50)
    print("  Sampling: RRDATASET")
    print("="*50)

    scenario_targets = {k: v for k, v in scenario_targets.items() if v > 0}
    n_real_total = scenario_targets.get(RR_REAL_BUCKET, 0)
    n_fake_total = sum(v for k, v in scenario_targets.items() if k != RR_REAL_BUCKET)

    ds = load_dataset(
        "alecrespi/RRDataset-CV-Project2",
        split="test",       # use the test split because it's the only split with actual jpg images in the jpg column
        streaming=True,
        token=hf_token,
    )
    ds = ds.shuffle(seed=seed, buffer_size=500)

    scenario_counts = {k: 0 for k in scenario_targets}
    real_count, fake_count, skip_count = 0, 0, 0
    saved_count = 0

    
    for entry in manifest.entries:
        if entry.get("source_dataset") != "rrdataset":
            continue
        bucket = RR_REAL_BUCKET if entry["label"] == "real" else entry.get("scenario")
        if bucket in scenario_counts:
            scenario_counts[bucket] += 1
        if entry["label"] == "real":
            real_count += 1
        elif entry["label"] == "fake":
            fake_count += 1

    def all_done():
        return all(scenario_counts[k] >= scenario_targets[k] for k in scenario_targets)

    def format_desc():
        return (f"RRDataset | real={real_count}/{n_real_total} "
                f"fake={fake_count}/{n_fake_total} skipped={skip_count}")

    progress = tqdm(ds, desc="RRDataset")
    for item in progress:
        if all_done():
            break

        key = item["__key__"]                  # e.g. ".../val/ai/redigital_War_&_Conflict_Scenes_000542"
        parts = key.split("/")
        label_folder = parts[-2]               # "ai" or "real"
        name_part = parts[-1]                  # e.g. "redigital_War_&_Conflict_Scenes_000542"

         
        scenario = next((s for s in RR_FAKE_SCENARIOS if s in name_part), None)
        if scenario is None:
            stripped = name_part[len("redigital_"):] if name_part.startswith("redigital_") else name_part
            tokens = stripped.rsplit("_", 1)
            scenario = tokens[0] if len(tokens) == 2 and tokens[1].isdigit() else stripped

        is_real = label_folder == "real"
        is_fake = label_folder == "ai"
        if not is_real and not is_fake:
            continue

        # bucket is "real" for real images, or the parsed scenario name for
        # fake images — only scenarios present in scenario_targets are kept.
        bucket = RR_REAL_BUCKET if is_real else scenario
        if bucket not in scenario_targets or scenario_counts[bucket] >= scenario_targets[bucket]:
            continue

        label_str = "real" if is_real else "fake"
        i = real_count if is_real else fake_count
        image_id = f"rrdataset_{label_str}_{i:02d}"
        filepath = output_dir / f"{image_id}.jpg"

        img_bytes = item["jpg"]

        try:
            if img_bytes is None:
                raise ValueError("missing image data")
            if isinstance(img_bytes, Image.Image):
                img = img_bytes
            elif isinstance(img_bytes, bytes): # some entries are already bytes, some are PIL images — handle both cases
                img = Image.open(io.BytesIO(img_bytes))
            else:
                img = Image.fromarray(img_bytes)
            save_image(img, filepath)
        except Exception as e:
            skip_count += 1
            progress.set_description(format_desc())
            continue

        manifest.add(make_entry(
            image_id=image_id,
            source="rrdataset",
            label=label_str,
            sensitivity="HIGH",
            scenario=scenario,
            original_label=label_str,
            hf_repo="alecrespi/RRDataset-CV-Project2",
            seed=seed,
        ))
        saved_count += 1
        scenario_counts[bucket] += 1

        if is_real:
            real_count += 1
        else:
            fake_count += 1

        progress.set_description(format_desc())

    print(f"  Scenario breakdown: {scenario_counts}")
    print(f"  Saved {saved_count} images")


# ──────────────────────────────────────────────────────────────
# DATASET 2 — DGM4
# Columns: id, text, image (file path STRING), fake_cls, ...
# Real: fake_cls == "orig"
# Tampered: any other value (face_swap, face_attribute, text_swap, etc.)
#           These are edits of real images, not fully synthetic —
#           labeled "tampered" rather than "fake".
# Image paths point to files inside the HF repo, which must be fetched
# ──────────────────────────────────────────────────────────────

def get_dgm4_image(img_path, zip_cache, hf_token):
    """
    To get extract the image file from the .zip in the HF repo
    img_path format: "DGM4/origin/bbc/0303/427.jpg"
    Zip file is at:  "DGM4/origin/bbc.zip"
    Inside zip:      "bbc/0303/427.jpg"
    """
    # e.g. ["DGM4", "origin", "bbc", "0303", "427.jpg"]
    parts = img_path.split("/")
    # zip path in repo: "origin/bbc.zip"  (no DGM4/ prefix in actual repo)
    zip_repo_path = "/".join(parts[1:3]) + ".zip"  # "origin/bbc.zip"
    # path inside zip: "bbc/0303/427.jpg"
    path_in_zip = "/".join(parts[2:])               # "bbc/0303/427.jpg"

    # Download the zip once, then cache it
    if zip_repo_path not in zip_cache:
        local_zip = hf_hub_download(
            repo_id="rshaojimmy/DGM4",
            filename=zip_repo_path,
            repo_type="dataset",
            token=hf_token,
        )
        zip_cache[zip_repo_path] = local_zip

    local_zip = zip_cache[zip_repo_path]

    import zipfile
    with zipfile.ZipFile(local_zip, "r") as zf:
        with zf.open(path_in_zip) as img_file:
            return Image.open(io.BytesIO(img_file.read()))


def sample_dgm4(output_dir, n_real, n_tampered, seed, hf_token, manifest):
    print("\n" + "="*50)
    print("  Sampling: DGM4")
    print("="*50)

    ds = load_dataset(
        "rshaojimmy/DGM4",
        split="test",
        streaming=True,
        token=hf_token,
    )
    ds = ds.shuffle(seed=seed, buffer_size=500)

    # Target counts per source, spread evenly for diversity
    real_quotas = distribute_evenly(n_real, ["bbc", "guardian", "usa_today", "washington_post"])
    tampered_quotas = distribute_evenly(n_tampered, ["infoswap", "simswap", "StyleCLIP", "HFGI"])
    real_counts = {k: 0 for k in real_quotas}
    tampered_counts = {k: 0 for k in tampered_quotas}

    skip_count = 0
    saved_count = 0
    zip_cache = {}

    def real_done():
        return all(real_counts[k] >= real_quotas[k] for k in real_quotas)

    def tampered_done():
        return all(tampered_counts[k] >= tampered_quotas[k] for k in tampered_quotas)

    def format_desc():
        r = sum(real_counts.values())
        t = sum(tampered_counts.values())
        return f"DGM4 | real={r}/{n_real} tampered={t}/{n_tampered} skipped={skip_count}"


    progress = tqdm(ds, desc="DGM4")
    for item in progress:
        if real_done() and tampered_done():
            break

        fake_cls = item["fake_cls"]
        is_real = fake_cls == "orig"
        img_path = item["image"]
        parts = img_path.split("/")  # e.g. ["DGM4", "origin", "bbc", "0303", "427.jpg"]

        if is_real:
            if real_done():
                continue
            # source is parts[2]: "bbc", "guardian", etc.
            source = parts[2]
            if source not in real_quotas or real_counts[source] >= real_quotas[source]:
                continue
        else:
            if tampered_done():
                continue
            # source is parts[2]: "infoswap", "simswap", etc.
            source = parts[2]
            if source not in tampered_quotas or tampered_counts[source] >= tampered_quotas[source]:
                continue

        label_str = "real" if is_real else "tampered"
        counts = real_counts if is_real else tampered_counts
        i = sum(counts.values())
        image_id = f"dgm4_{label_str}_{i:02d}"
        filepath = output_dir / f"{image_id}.jpg"

        try:
            img = get_dgm4_image(img_path, zip_cache, hf_token)
            save_image(img, filepath)
        except Exception as e:
            skip_count += 1
            progress.set_description(format_desc())
            continue

        manifest.add(make_entry(
            image_id=image_id,
            source="dgm4",
            label=label_str,
            sensitivity="MEDIUM-HIGH",
            scenario="news_politics",
            generator=item["fake_cls"] if label_str == "tampered" else None,
            original_label=item["fake_cls"],
            hf_repo="rshaojimmy/DGM4",
            seed=seed,
        ))
        saved_count += 1

        counts[source] += 1
        progress.set_description(format_desc())

    print(f"  Real collected: { {k: v for k,v in real_counts.items()} }")
    print(f"  Tampered collected: { {k: v for k,v in tampered_counts.items()} }")
    print(f"  Saved {saved_count} images")


# ──────────────────────────────────────────────────────────────
# DATASET 3 — SID-Set
# Columns: img_id, image (PIL), mask, label (int)
# Real: label == 0 | Fake (full_synthetic): label == 1 | Tampered: label == 2
# ──────────────────────────────────────────────────────────────

def sample_sidset(output_dir, n_real, n_fake, n_tampered, seed, hf_token, manifest):
    print("\n" + "="*50)
    print("  Sampling: SID-Set")
    print("="*50)

    ds = load_dataset(
        "saberzl/SID_Set",
        split="validation",
        streaming=True,
        token=hf_token,
    )
    ds = ds.shuffle(seed=seed, buffer_size=500)

    real_count, fake_count, tampered_count, skip_count = 0, 0, 0, 0
    saved_count = 0

    def counts_for(label_int):
        if label_int == 0:
            return "real", real_count, n_real
        elif label_int == 1:
            return "fake", fake_count, n_fake
        else:  # label_int == 2
            return "tampered", tampered_count, n_tampered

    def format_desc():
        return (f"SID-Set | real={real_count}/{n_real} fake={fake_count}/{n_fake} "
                f"tampered={tampered_count}/{n_tampered} skipped={skip_count}")

    progress = tqdm(ds, desc="SID-Set")
    for item in progress:
        if real_count >= n_real and fake_count >= n_fake and tampered_count >= n_tampered:
            break

        label_int = item["label"]
        label_str, current, target = counts_for(label_int)

        if current >= target:
            continue

        i = current
        image_id = f"sidset_{label_str}_{i:02d}"
        filepath = output_dir / f"{image_id}.jpg"

        try:
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            save_image(img, filepath)
        except Exception as e:
            skip_count += 1
            progress.set_description(format_desc())
            continue

        manifest.add(make_entry(
            image_id=image_id,
            source="sidset",
            label=label_str,
            sensitivity="LOW",
            scenario="social_media",
            original_label=str(item["label"]),
            hf_repo="saberzl/SID_Set",
            seed=seed,
        ))
        saved_count += 1

        if label_str == "real":
            real_count += 1
        elif label_str == "fake":
            fake_count += 1
        else:
            tampered_count += 1

        progress.set_description(format_desc())

    print(f"  Saved {saved_count} images")


# ──────────────────────────────────────────────────────────────
# DATASET 4 — Sens-VisualNews
# Sourced from the base VisualNews corpus. 
# All VisualNews images are real, unaltered news photos, so this dataset
# only ever contributes to the "real" label. VisualNews itself carries no
# sensationalism/CSS score, so we bias the sample toward the kind of
# content likely to land at CSS level 3-4 in Phase 2 scoring — war &
# conflict, mass casualties, weapons — via keyword matching against each
# entry's caption, bucketed for diversity across those categories.
#
# data.json entries look like:
#   {"id": 39136, "caption": "...", "topic": "law_crime",
#    "source": "washington_post",
#    "image_path": "./washington_post/images/0376/501.jpg",
#    "article_path": "./washington_post/articles/39136.txt"}
# ──────────────────────────────────────────────────────────────

SVN_KEYWORD_BUCKETS = {
    "war_conflict": [
        "war", "wars", "warfare", "battle", "battles", "battlefield",
        "combat", "airstrike", "airstrikes", "air strike", "air strikes",
        "shelling", "bombing", "bombings", "bombed", "gunfire", "firefight",
        "firefights", "insurgent", "insurgents", "militant", "militants",
        "rebel fighters", "offensive", "invasion",
    ],
    "mass_casualty": [
        "killed", "dead", "death toll", "casualty", "casualties", "corpse",
        "corpses", "bodies", "massacre", "massacres", "genocide",
        "fatality", "fatalities", "dying", "slain", "mass grave",
        "mass graves",
    ],
    "weapons": [
        "gun", "guns", "rifle", "rifles", "firearm", "firearms", "grenade",
        "grenades", "explosive", "explosives", "bomb", "bombs", "missile",
        "missiles", "artillery", "weapon", "weapons", "ammunition",
        "gunman", "gunmen", "shooter", "shooters",
    ],
    "graphic_violence": [
        "blood", "bloodied", "wounded", "injured", "mutilated", "torture",
        "tortured", "execution", "executions", "beheaded", "atrocity",
        "atrocities",
    ],
}

# Precompile regex patterns for each bucket for efficiency
_SVN_BUCKET_PATTERNS = {
    bucket: re.compile(
        r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
    )
    for bucket, keywords in SVN_KEYWORD_BUCKETS.items()
}


def _classify_svn_bucket(caption):
    """
    Return the first keyword bucket (in SVN_KEYWORD_BUCKETS order) whose
    terms appear in `caption` as whole words (case-insensitive), or None if
    the caption doesn't match any disturbing-content keyword.
    """
    if not caption:
        return None
    text = caption.lower()
    for bucket, pattern in _SVN_BUCKET_PATTERNS.items():
        if pattern.search(text):
            return bucket
    return None


def sample_sens_visualnews(output_dir, n_real, seed, tar_path, manifest):
    """
    n_real: total number of images to pull, split evenly across
    SVN_KEYWORD_BUCKETS for diversity (mirrors how sample_dgm4 splits its
    quotas evenly across sub-sources).
    """
    print("\n" + "="*50)
    print("  Sampling: SENS-VISUALNEWS")
    print("="*50)

    if n_real <= 0:
        print("  Nothing targeted for Sens-VisualNews — skipping")
        return

    if not tar_path or not Path(tar_path).exists():
        print(f"  Skipping: origin.tar not found at {tar_path!r} "
              f"(pass --sens_visualnews_tar /path/to/origin.tar)")
        return

    tf = tarfile.open(tar_path, mode="r:")

    print("  Indexing tar (single pass over headers only, no image data read)...")
    all_members = tf.getmembers()
    members_by_name = {m.name: m for m in all_members}
    print(f"  Indexed {len(members_by_name)} entries")

    data_json_name = next((n for n in members_by_name if n.endswith("data.json")), None)
    if data_json_name is None:
        print("  Skipping: couldn't find data.json inside the tar")
        tf.close()
        return
    prefix = data_json_name[: -len("data.json")]  # e.g. "" or "origin/"

    with tf.extractfile(members_by_name[data_json_name]) as f:
        entries = json.load(f)
    print(f"  Loaded {len(entries)} VisualNews entries")

    # Bucket every entry by disturbing-content keyword match in its caption.
    buckets = {k: [] for k in SVN_KEYWORD_BUCKETS}
    for entry in entries:
        bucket = _classify_svn_bucket(entry.get("caption"))
        if bucket is not None:
            buckets[bucket].append(entry)

    rng = random.Random(seed)
    for bucket_entries in buckets.values():
        rng.shuffle(bucket_entries)

    for bucket, bucket_entries in buckets.items():
        print(f"  {bucket}: {len(bucket_entries)} candidates")

    bucket_targets = distribute_evenly(n_real, list(SVN_KEYWORD_BUCKETS.keys()))
    counts = {k: 0 for k in SVN_KEYWORD_BUCKETS}
    saved_count, skip_count = 0, 0

    def format_desc():
        return f"Sens-VisualNews | {counts} skipped={skip_count}"

    progress = tqdm(total=sum(bucket_targets.values()), desc="Sens-VisualNews")
    for bucket, target in bucket_targets.items():
        if target <= 0:
            continue
        for entry in buckets[bucket]:
            if counts[bucket] >= target:
                break

            image_path = entry.get("image_path", "").lstrip("./")
            member = members_by_name.get(prefix + image_path)
            if member is None:
                skip_count += 1
                progress.set_description(format_desc())
                continue

            image_id = f"sensvisualnews_real_{saved_count:02d}"
            filepath = output_dir / f"{image_id}.jpg"

            try:
                with tf.extractfile(member) as imgf:
                    img = Image.open(io.BytesIO(imgf.read()))
                save_image(img, filepath)
            except Exception:
                skip_count += 1
                progress.set_description(format_desc())
                continue

            manifest.add(make_entry(
                image_id=image_id,
                source="sensvisualnews",
                label="real",
                sensitivity="HIGH",
                scenario=bucket,
                original_label=entry.get("topic"),
                hf_repo=None,
                seed=seed,
            ))
            saved_count += 1
            counts[bucket] += 1
            progress.update(1)
            progress.set_description(format_desc())

    progress.close()
    tf.close()
    print(f"  Bucket breakdown: {counts}")
    print(f"  Saved {saved_count} images (skipped {skip_count})")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def already_done(output_dir, prefix, label_counts):
    """Check if a dataset was already fully collected in a previous run.

    `label_counts` maps label -> required count, e.g. {"real": 10, "fake": 10}
    or {"real": 10, "fake": 5, "tampered": 5} for datasets with a 3-way split.
    """
    return all(
        len(list(output_dir.glob(f"{prefix}_{label}_*.jpg"))) >= count
        for label, count in label_counts.items()
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Sample images")
    parser.add_argument("--output_dir",    type=str, default="/mnt/ssd1/bairat/dataset/dataset3")
    parser.add_argument("--metadata_path", type=str, default="/mnt/ssd1/bairat/metadata/dataset3/manifest.json")
    parser.add_argument("--hf_token",      type=str, default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token (default: $HF_TOKEN from the environment)")
    parser.add_argument("--seed",          type=int, default=42)

    # How many images to pull from each (dataset, label) pair. 
    parser.add_argument("--n_rrdataset_real", type=int, default=0,
                        help="RRDataset real images")
    parser.add_argument("--n_rrdataset_fake", type=int, default=0,
                        help="RRDataset fake images — split evenly across the 5 fake scenarios")
    parser.add_argument("--n_dgm4_real", type=int, default=0,
                        help="DGM4 real images")
    parser.add_argument("--n_dgm4_tampered", type=int, default=0,
                        help="DGM4 tampered images")
    parser.add_argument("--n_sidset_real", type=int, default=0,
                        help="SID-Set real images")
    parser.add_argument("--n_sidset_fake", type=int, default=0,
                        help="SID-Set fake images")
    parser.add_argument("--n_sidset_tampered", type=int, default=0,
                        help="SID-Set tampered images")
    parser.add_argument("--n_sensvisualnews_real", type=int, default=0,
                        help="Sens-VisualNews real images (default: 0, i.e. opt-in — "
                             "set > 0 and pass --sens_visualnews_tar to use it)")

    # Sens-VisualNews is read directly out of a local VisualNews origin.tar
    # (91GB, downloaded separately from https://www.cs.rice.edu/~vo9/visualnews/ 
    parser.add_argument("--sens_visualnews_tar", type=str, default="/mnt/ssd1/bairat/dataset/sens-visualnews/origin.tar",
                        help="Local path to VisualNews' origin.tar. Required if "
                             "--n_sensvisualnews_real > 0.")

    # Manual skip flags (useful when some datasets are unavailable or already collected)
    parser.add_argument("--skip_rrdataset", action="store_true",
                        help="Force skip RRDataset")
    parser.add_argument("--skip_dgm4",      action="store_true",
                        help="Force skip DGM4")
    parser.add_argument("--skip_sidset",    action="store_true",
                        help="Force skip SID-Set")
    parser.add_argument("--skip_sensvisualnews", action="store_true",
                        help="Force skip Sens-VisualNews")
    args = parser.parse_args()

    hf_token = args.hf_token

    # Per-(dataset, label) counts; each number is exactly what gets sampled for
    # that label
    label_counts = {
        "rrdataset": {"real": args.n_rrdataset_real, "fake": args.n_rrdataset_fake},
        "dgm4": {"real": args.n_dgm4_real, "tampered": args.n_dgm4_tampered},
        "sidset": {"real": args.n_sidset_real, "fake": args.n_sidset_fake, "tampered": args.n_sidset_tampered},
        "sensvisualnews": {"real": args.n_sensvisualnews_real},
    }
    if any(v < 0 for counts in label_counts.values() for v in counts.values()):
        parser.error("all --n_* counts must be >= 0")
    dataset_totals = {name: sum(counts.values()) for name, counts in label_counts.items()}
    total_images = sum(dataset_totals.values())
    if total_images <= 0:
        parser.error("all --n_* counts can't be 0")

    # RRDataset's fake count is the only one still split further — evenly
    # across the 5 fake scenarios, since there's no per-scenario argument.
    rr_scenario_targets = distribute_evenly(label_counts["rrdataset"]["fake"], RR_FAKE_SCENARIOS)
    rr = {RR_REAL_BUCKET: label_counts["rrdataset"]["real"], **rr_scenario_targets}
    dgm4 = label_counts["dgm4"]
    sidset = label_counts["sidset"]
    svn = label_counts["sensvisualnews"]

    label_grand_totals = {"real": 0, "fake": 0, "tampered": 0}
    for counts in label_counts.values():
        for label, v in counts.items():
            label_grand_totals[label] += v

    print("Dataset image counts:")
    for name, counts in label_counts.items():
        n = dataset_totals[name]
        pct = 100 * n / total_images
        breakdown = ", ".join(f"{label}={v}" for label, v in counts.items())
        print(f"  {name:16s}: {n:8d} images ({pct:5.1f}%)  [{breakdown}]")
    print(f"  {'TOTAL':16s}: {total_images:8d} images")

    print("\nLabel totals (across all datasets):")
    for label in ("real", "fake", "tampered"):
        n = label_grand_totals[label]
        pct = 100 * n / total_images
        print(f"  {label:9s}: {n:8d} images ({pct:5.1f}%)")

    print("\nRRDataset scenario breakdown:")
    for bucket, n in rr.items():
        print(f"  {bucket:30s}: {n:8d} images")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)

    # ManifestWriter loads any existing manifest and flushes to disk after
    # every single image added, so a crash mid-run only loses at most the
    # image currently in flight — never previously-saved metadata.
    manifest = ManifestWriter(args.metadata_path)
    print(f"Loaded {len(manifest)} existing entries from manifest")

    # ── RRDataset ──────────────────────────────────────────────
    # already_done() checks by label count, not by scenario, so we sum the 5 fake scenarios into a single "fake" count for that check.
    rr_label_totals = {
        "real": rr.get(RR_REAL_BUCKET, 0),
        "fake": sum(v for k, v in rr.items() if k != RR_REAL_BUCKET),
    }
    if args.skip_rrdataset:
        print("\nSkipping RRDataset (--skip_rrdataset)")
    elif already_done(output_dir, "rrdataset", rr_label_totals):
        print("\nRRDataset already complete — skipping (delete images to re-sample)")
    else:
        sample_rrdataset(output_dir, rr, args.seed, hf_token, manifest)

    # ── DGM4 ───────────────────────────────────────────────────
    if args.skip_dgm4:
        print("\nSkipping DGM4 (--skip_dgm4)")
    elif already_done(output_dir, "dgm4", dgm4):
        print("\nDGM4 already complete — skipping (delete images to re-sample)")
    else:
        sample_dgm4(output_dir, dgm4["real"], dgm4["tampered"], args.seed, hf_token, manifest)

    # ── SID-Set ────────────────────────────────────────────────
    if args.skip_sidset:
        print("\nSkipping SID-Set (--skip_sidset)")
    elif already_done(output_dir, "sidset", sidset):
        print("\nSID-Set already complete — skipping (delete images to re-sample)")
    else:
        sample_sidset(output_dir, sidset["real"], sidset["fake"], sidset["tampered"], args.seed, hf_token, manifest)

    # ── Sens-VisualNews ────────────────────────────────────────
    svn_label_totals = {"real": svn.get("real", 0)}
    if args.skip_sensvisualnews:
        print("\nSkipping Sens-VisualNews (--skip_sensvisualnews)")
    elif svn_label_totals["real"] <= 0:
        print("\nSkipping Sens-VisualNews (0 images targeted — set --n_sensvisualnews_real > 0)")
    elif already_done(output_dir, "sensvisualnews", svn_label_totals):
        print("\nSens-VisualNews already complete — skipping (delete images to re-sample)")
    else:
        sample_sens_visualnews(output_dir, svn["real"], args.seed, args.sens_visualnews_tar, manifest)

    print("\n" + "="*50)
    print(f"  DONE — {len(manifest)} images collected")
    print(f"  Images   : {args.output_dir}")
    print(f"  Manifest : {args.metadata_path}")
    print("="*50)

    sys.exit(0)


if __name__ == "__main__":
    main()