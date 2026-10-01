#!/usr/bin/env python3
"""Build a UC-VLM visual-adaptation dataset in LLaMA-Factory ShareGPT format."""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with: pip install Pillow") from exc


DEFAULT_PROMPT = (
    "<image>Classify whether this image is AI-generated or authentic. "
    "Answer only with authentic or generated. No explanations."
)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan authentic/generated image directories and create a balanced "
            "multimodal ShareGPT JSON dataset for UC-VLM visual adaptation."
        )
    )
    parser.add_argument("--real-dir", type=Path, required=True, help="Directory containing authentic images.")
    parser.add_argument(
        "--generated-dir", type=Path, required=True, help="Directory containing AI-generated images."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file.")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=18000,
        help="Maximum valid images per class; use 0 to include all images (default: 18000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and shuffling.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt stored in every example.")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip Pillow image verification. Faster, but corrupt images may enter the dataset.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing an existing output file atomically."
    )
    return parser.parse_args()


def discover_images(root):
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def is_valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def select_images(root, limit, verify, rng):
    candidates = discover_images(root)
    rng.shuffle(candidates)
    selected = []
    invalid = []
    seen = set()

    for path in candidates:
        normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)

        if verify and not is_valid_image(path):
            invalid.append(path)
            continue

        selected.append(path)
        if limit and len(selected) >= limit:
            break

    return selected, invalid, len(candidates)


def make_record(image_path, label, prompt):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label},
        ],
        "images": [str(image_path)],
    }


def validate_inputs(args):
    for name, directory in (("real", args.real_dir), ("generated", args.generated_dir)):
        if not directory.is_dir():
            raise ValueError("{} image directory does not exist: {}".format(name, directory))

    if args.limit_per_class < 0:
        raise ValueError("--limit-per-class must be 0 or a positive integer.")

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            "Output already exists: {}. Pass --overwrite to replace it.".format(args.output)
        )


def main():
    args = parse_args()
    try:
        validate_inputs(args)
    except (ValueError, FileExistsError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    verify = not args.no_verify
    real_images, invalid_real, real_candidates = select_images(
        args.real_dir, args.limit_per_class, verify, rng
    )
    generated_images, invalid_generated, generated_candidates = select_images(
        args.generated_dir, args.limit_per_class, verify, rng
    )

    records = [make_record(path, "authentic", args.prompt) for path in real_images]
    records.extend(make_record(path, "generated", args.prompt) for path in generated_images)
    rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_output.replace(args.output)

    print("Dataset written to: {}".format(args.output.resolve()))
    print("Authentic: {}/{} candidates".format(len(real_images), real_candidates))
    print("Generated: {}/{} candidates".format(len(generated_images), generated_candidates))
    print("Invalid images skipped: {}".format(len(invalid_real) + len(invalid_generated)))
    print("Total records: {}".format(len(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
