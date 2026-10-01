#!/usr/bin/env python3
"""Merge the UC-VLM Stage-1 visual LoRA into the Qwen2.5-VL backbone."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a visual LoRA adapter into a Qwen2.5-VL backbone.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Stage-1 LoRA directory containing adapter_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/tanlei/UC-VLM/UC-VLM/output/qwen2_5vl_visual_merged"),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Weight dtype used while loading and saving the merged model.",
    )
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def validate_args(args):
    if not args.adapter_path.is_dir():
        raise FileNotFoundError("Adapter directory not found: {}".format(args.adapter_path))
    if not (args.adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError("adapter_config.json not found in: {}".format(args.adapter_path))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Output directory is not empty: {}. Choose a new directory to avoid mixing model files.".format(
                args.output_dir
            )
        )


def main():
    args = parse_args()
    try:
        validate_args(args)
    except (FileNotFoundError, FileExistsError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except (ImportError, AttributeError) as exc:
        print(
            "error: use the UC-VLM Python 3.11+ environment with current torch, transformers, and peft.",
            file=sys.stderr,
        )
        return 2

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print("[MERGE] Loading processor: {}".format(args.base_model))
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    print("[MERGE] Loading backbone in {}: {}".format(args.dtype, args.base_model))
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("[MERGE] Loading visual LoRA: {}".format(args.adapter_path))
    peft_model = PeftModel.from_pretrained(backbone, str(args.adapter_path))
    print("[MERGE] Merging adapter weights into the backbone.")
    try:
        merged_model = peft_model.merge_and_unload(safe_merge=True)
    except TypeError:
        # Compatibility with older PEFT releases that do not expose safe_merge.
        merged_model = peft_model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[MERGE] Saving merged model: {}".format(args.output_dir))
    merged_model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(args.output_dir)

    # Confirm that the saved directory is a loadable Transformers model before reporting success.
    saved_config = AutoConfig.from_pretrained(args.output_dir, trust_remote_code=True)
    model_files = sorted(path.name for path in args.output_dir.glob("*.safetensors"))
    if not model_files and not (args.output_dir / "model.safetensors.index.json").is_file():
        print("error: no safetensors weights were written.", file=sys.stderr)
        return 1

    manifest = {
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "dtype": args.dtype,
        "model_type": getattr(saved_config, "model_type", None),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (args.output_dir / "merge_manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")

    print("[MERGE] Completed successfully.")
    print("[MERGE] Use this model path: {}".format(args.output_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
