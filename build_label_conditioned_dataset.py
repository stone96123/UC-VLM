#!/usr/bin/env python3
"""Build the UC-VLM label-conditioned regeneration dataset (Stage 3)."""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate assistant reasoning with a manually supplied or evolved prompt. "
            "Wrong initial predictions are regenerated with a ground-truth label directive."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Optional Stage-1 visual LoRA directory. Omit when --model-name is already a merged model.",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("/home/tanlei/UC-VLM/UC-VLM/data/only_sdv4_vt_11.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/tanlei/UC-VLM/UC-VLM/data/only_sdv4_label_conditioned.json"),
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="Manually enter the complete prompt as a command-line string.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Read a manually written prompt from a UTF-8 file.")
    prompt_group.add_argument(
        "--prompt-result",
        type=Path,
        help="Read the selected prompt from best_prompt.json or final_evolution_result.json.",
    )

    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=3000,
        help="Maximum source examples per class; use 0 for all records (default: 3000).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--regeneration-attempts",
        type=int,
        default=2,
        help="Maximum label-conditioned retries after an incorrect initial prediction.",
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--keep-failed-regeneration",
        action="store_true",
        help="Keep the last answer even when its final verdict still disagrees with the label.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output JSON.")
    return parser.parse_args()


def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in {}: {}".format(path, exc))


def extract_prompt(payload):
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        raise ValueError("Prompt result must be a JSON object or string.")
    if isinstance(payload.get("prompt"), str):
        return payload["prompt"].strip()
    if isinstance(payload.get("best"), dict) and isinstance(payload["best"].get("prompt"), str):
        return payload["best"]["prompt"].strip()
    raise ValueError("Could not find `prompt` or `best.prompt` in the prompt result.")


def read_interactive_prompt():
    if not sys.stdin.isatty():
        raise ValueError("Provide --prompt, --prompt-file, or --prompt-result in non-interactive mode.")
    print("Paste the complete prompt. Enter a line containing only END when finished:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def resolve_prompt(args):
    if args.prompt is not None:
        prompt = args.prompt.strip()
    elif args.prompt_file is not None:
        if not args.prompt_file.is_file():
            raise FileNotFoundError("Prompt file not found: {}".format(args.prompt_file))
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    elif args.prompt_result is not None:
        if not args.prompt_result.is_file():
            raise FileNotFoundError("Prompt result not found: {}".format(args.prompt_result))
        prompt = extract_prompt(read_json(args.prompt_result))
    else:
        prompt = read_interactive_prompt()

    prompt = prompt.replace("<image>", "").strip()
    if not prompt:
        raise ValueError("The supplied prompt is empty.")
    return prompt


def normalize_label(value):
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value in {"authentic", "real"}:
        return "Authentic"
    if value in {"generated", "fake", "ai-generated", "ai generated"}:
        return "Generated"
    return None


def extract_verdict(answer):
    matches = list(
        re.finditer(
            r"(?is)(?:\[\s*step\s*[-_]?\s*4\s*\]|step\s*[-_]?\s*4|final\s+assessment)",
            answer,
        )
    )
    segment = answer[matches[-1].start() :] if matches else answer
    verdicts = re.findall(r"(?i)\b(ai[-\s]?generated|generated|authentic|fake|real)\b", segment)
    return normalize_label(verdicts[0]) if verdicts else None


def build_label_directive(label, prompt):
    if label == "Generated":
        directive = (
            "This image is 'Generated' (AI-generated). Explain why this image is AI-generated, not authentic. "
            "Your final verdict must be 'Generated'."
        )
    else:
        directive = (
            "This image is 'Authentic' (not AI-generated). Explain why this image is authentic, not AI-generated. "
            "Your final verdict must be 'Authentic'."
        )
    return directive + "\n\n" + prompt


def load_source_examples(path, limit_per_class, rng):
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("Source dataset must contain a JSON list.")

    grouped = {"Authentic": [], "Generated": []}
    skipped = 0
    seen = set()
    for record in payload:
        try:
            image_path = str(Path(record["images"][0]).resolve())
            assistant_messages = [message for message in record["messages"] if message.get("role") == "assistant"]
            label = normalize_label(assistant_messages[-1]["content"])
        except (KeyError, IndexError, TypeError, AttributeError):
            skipped += 1
            continue
        if label is None or not Path(image_path).is_file() or (image_path, label) in seen:
            skipped += 1
            continue
        seen.add((image_path, label))
        grouped[label].append((image_path, label))

    selected = []
    for label in ("Authentic", "Generated"):
        rng.shuffle(grouped[label])
        selected.extend(grouped[label][:limit_per_class] if limit_per_class else grouped[label])
    rng.shuffle(selected)
    if not selected:
        raise ValueError("No valid examples were found in the source dataset.")
    print(
        "[DATA] Selected {} authentic and {} generated examples; skipped {} invalid records.".format(
            sum(label == "Authentic" for _, label in selected),
            sum(label == "Generated" for _, label in selected),
            skipped,
        )
    )
    return selected


class Generator:
    def __init__(self, model_name, adapter_path, image_size, max_new_tokens):
        try:
            import torch
            from peft import PeftModel
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Use the UC-VLM Python 3.11+ environment with torch, transformers, peft, and qwen-vl-utils."
            ) from exc

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.image_size = image_size
        self.max_new_tokens = max_new_tokens
        print("[MODEL] Loading processor: {}".format(model_name))
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("[MODEL] Loading model: {}".format(model_name))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path is not None:
            print("[MODEL] Loading visual adapter: {}".format(adapter_path))
            model = PeftModel.from_pretrained(model, str(adapter_path))
        self.model = model.eval()

    def generate(self, image_path, prompt):
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST
                )
        except (UnidentifiedImageError, OSError, ValueError):
            return None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def make_record(image_path, prompt, answer):
    return {
        "messages": [
            {"role": "user", "content": "<image>" + prompt},
            {"role": "assistant", "content": answer.strip()},
        ],
        "images": [image_path],
    }


def write_json_atomic(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary.replace(path)


def validate_args(args):
    if not args.source_dataset.is_file():
        raise FileNotFoundError("Source dataset not found: {}".format(args.source_dataset))
    if args.adapter_path is not None:
        if not args.adapter_path.is_dir() or not (args.adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError("Invalid adapter directory: {}".format(args.adapter_path))
    if args.limit_per_class < 0:
        raise ValueError("--limit-per-class must be 0 or a positive integer.")
    if args.image_size < 1 or args.max_new_tokens < 1 or args.regeneration_attempts < 1 or args.save_every < 1:
        raise ValueError("Image size, token count, attempts, and save interval must be positive.")
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError("Output exists: {}. Use --resume or --overwrite.".format(args.output))


def main():
    args = parse_args()
    try:
        validate_args(args)
        prompt = resolve_prompt(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    if "no explanations" in prompt.lower() or "answer only" in prompt.lower():
        print(
            "[WARN] The prompt requests a label-only answer. Use a reasoning prompt if reconstructed content is required.",
            file=sys.stderr,
        )

    rng = random.Random(args.seed)
    try:
        examples = load_source_examples(args.source_dataset, args.limit_per_class, rng)
        generator = Generator(args.model_name, args.adapter_path, args.image_size, args.max_new_tokens)
    except (ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    records = []
    if args.resume and args.output.exists():
        existing = read_json(args.output)
        if not isinstance(existing, list):
            print("error: resume output is not a JSON list.", file=sys.stderr)
            return 2
        records.extend(existing)
    processed_paths = {record["images"][0] for record in records if record.get("images")}

    stats = Counter()
    for index, (image_path, label) in enumerate(examples, 1):
        if image_path in processed_paths:
            stats["already_processed"] += 1
            continue

        answer = generator.generate(image_path, prompt)
        if not answer:
            stats["invalid_image_or_empty"] += 1
            continue

        prediction = extract_verdict(answer)
        if prediction == label:
            stats["initially_correct"] += 1
        else:
            directive_prompt = build_label_directive(label, prompt)
            corrected = False
            for _ in range(args.regeneration_attempts):
                regenerated_answer = generator.generate(image_path, directive_prompt)
                if regenerated_answer:
                    answer = regenerated_answer
                if extract_verdict(answer) == label:
                    corrected = True
                    break
            if corrected:
                stats["regenerated_correctly"] += 1
            elif args.keep_failed_regeneration:
                stats["kept_failed_regeneration"] += 1
            else:
                stats["dropped_failed_regeneration"] += 1
                continue

        records.append(make_record(image_path, prompt, answer))
        processed_paths.add(image_path)
        if len(records) % args.save_every == 0:
            write_json_atomic(args.output, records)
            print("[SAVE] {} records written; progress {}/{}.".format(len(records), index, len(examples)))

    write_json_atomic(args.output, records)
    print("Dataset written to: {}".format(args.output.resolve()))
    print("Total records: {}".format(len(records)))
    print("Statistics: {}".format(dict(stats)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
