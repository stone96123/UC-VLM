#!/usr/bin/env python3
"""Stage 2 of UC-VLM: accuracy-guided prompt evolution.

This implementation follows Algorithm 1 in the paper:
1. Generate an initial prompt pool with an LLM.
2. Score every prompt on a fixed, balanced validation subset.
3. Retain the Top-k prompts.
4. Refill the pool with LLM-generated children using rewrite/modify with
   equal probability.
5. Repeat for M rounds and save the globally best prompt.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


BASIC_PROMPT = (
    "Classify whether this image is AI-generated or authentic. "
    "Answer only with authentic or generated. No explanations."
)

TASK_LINE = "Classify whether this image is AI-generated or authentic."
FINAL_STEP = (
    "[Step-4] Final Assessment: Give one word stating 'Generated' or 'Authentic'. "
    "Summarize the key evidence."
)


BASE_PROMPT = """Classify whether this image is AI-generated or authentic.
Follow the four steps below strictly and preserve their headings exactly as written.
[Step-1] Visual Examination: List up to three suspicious and three realistic visual cues, such as glitches, lighting, texture, or motion blur.
[Step-2] Statistical Analysis: Highlight low-level or frequency-domain evidence supporting or contradicting authenticity.
[Step-3] Semantic & Physics Consistency: Identify impossible reflections, shadow errors, geometry problems, depth inconsistencies, or physical mismatches.
[Step-4] Final Assessment: Give one word stating 'Generated' or 'Authentic'. Summarize the key evidence."""


@dataclass
class EvolutionConfig:
    model_name: str
    adapter_path: str
    dataset_json: str
    output_dir: str
    openai_model: str
    pool_size: int
    top_k: int
    rounds: int
    samples_per_class: int
    seed: int
    max_new_tokens: int
    openai_max_tokens: int
    openai_temperature: float


def parse_args():
    parser = argparse.ArgumentParser(description="Run UC-VLM Stage-2 prompt evolution.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Optional Stage-1 visual LoRA. Omit when --model-name points to an already merged model.",
    )
    parser.add_argument("--dataset-json", type=Path, required=True, help="ShareGPT validation dataset JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/tanlei/UC-VLM/prompt_evolution_outputs"),
    )
    parser.add_argument("--openai-model", default="gpt-4o", help="LLM used to evolve prompts.")
    parser.add_argument("--pool-size", type=int, default=5, help="Instruction pool size N.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of elite prompts retained each round.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of evolution rounds M.")
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--openai-max-tokens", type=int, default=500)
    parser.add_argument("--openai-temperature", type=float, default=0.7)
    return parser.parse_args()


def validate_args(args):
    if args.pool_size < 2:
        raise ValueError("--pool-size must be at least 2.")
    if not 0 < args.top_k < args.pool_size:
        raise ValueError("--top-k must satisfy 0 < top-k < pool-size.")
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1.")
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1.")
    if not args.dataset_json.is_file():
        raise FileNotFoundError("Dataset JSON not found: {}".format(args.dataset_json))
    if args.adapter_path is not None:
        if not args.adapter_path.is_dir():
            raise FileNotFoundError("Stage-1 adapter directory not found: {}".format(args.adapter_path))
        if not (args.adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError("adapter_config.json not found in: {}".format(args.adapter_path))
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running prompt evolution.")


def normalize_label(value):
    value = value.strip().lower()
    if value in {"authentic", "real"}:
        return "Authentic"
    if value in {"generated", "fake", "ai-generated", "ai generated"}:
        return "Generated"
    return None


def load_balanced_validation_set(dataset_path, samples_per_class, rng):
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        records = json.load(dataset_file)

    grouped = {"Authentic": [], "Generated": []}
    missing_images = 0
    malformed_records = 0
    seen = set()

    for record in records:
        try:
            image_path = Path(record["images"][0]).resolve()
            assistant_messages = [m for m in record["messages"] if m.get("role") == "assistant"]
            label = normalize_label(assistant_messages[-1]["content"])
        except (KeyError, IndexError, TypeError, AttributeError):
            malformed_records += 1
            continue

        if label is None:
            malformed_records += 1
            continue
        if not image_path.is_file():
            missing_images += 1
            continue

        key = (str(image_path), label)
        if key in seen:
            continue
        seen.add(key)
        grouped[label].append((str(image_path), label))

    for examples in grouped.values():
        rng.shuffle(examples)

    sample_count = min(samples_per_class, len(grouped["Authentic"]), len(grouped["Generated"]))
    if sample_count == 0:
        raise ValueError("The dataset does not contain usable examples from both classes.")

    selected = grouped["Authentic"][:sample_count] + grouped["Generated"][:sample_count]
    rng.shuffle(selected)
    print(
        "[DATA] Using {} authentic + {} generated images; skipped {} missing and {} malformed records.".format(
            sample_count, sample_count, missing_images, malformed_records
        )
    )
    return selected


def clean_prompt(text):
    text = text.strip()
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def is_valid_prompt(prompt):
    lowered = prompt.lower()
    headings = ("[step-1]", "[step-2]", "[step-3]", "[step-4]")
    positions = [lowered.find(heading) for heading in headings]
    return (
        prompt.startswith(TASK_LINE)
        and all(lowered.count(heading) == 1 for heading in headings)
        and positions == sorted(positions)
        and FINAL_STEP in prompt
    )


class PromptGenerator:
    def __init__(self, model, temperature, max_tokens, rng):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI SDK with: pip install openai") from exc

        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rng = rng

    def _call(self, instruction):
        last_error = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": "You design concise prompts for AI-generated image forensic classification.",
                        },
                        {"role": "user", "content": instruction},
                    ],
                )
                candidate = clean_prompt(response.choices[0].message.content or "")
                if is_valid_prompt(candidate):
                    return candidate
                last_error = ValueError("The generated prompt does not contain all four required steps.")
            except Exception as exc:  # API and transport errors are retried together.
                last_error = exc

            if attempt < 2:
                time.sleep(2**attempt)

        raise RuntimeError("Prompt generation failed after 3 attempts: {}".format(last_error))

    def initialize(self, base_prompt):
        instruction = """Create ONE instruction variant for authentic-versus-AI-generated image classification.

Requirements:
- Begin with this exact first line: {task_line}
- Preserve exactly one occurrence of each heading, in this order: [Step-1], [Step-2], [Step-3], [Step-4].
- Cover visual evidence, low-level/statistical evidence, and semantic/physics consistency.
- Keep this final step verbatim: {final_step}
- Change wording or forensic emphasis so the candidate is meaningfully distinct.
- Do not mention datasets, generator names, class priors, accuracy, or a preferred verdict.
- Do not state or imply the correct label before inspecting the image.
- Output only the candidate instruction, without commentary or code fences.
- Keep it below 250 words.

BASE INSTRUCTION
----------------
{base_prompt}
----------------""".format(task_line=TASK_LINE, final_step=FINAL_STEP, base_prompt=base_prompt)
        return self._call(instruction)

    def mutate(self, parent_prompt):
        mode = "rewrite" if self.rng.random() < 0.5 else "modify"
        if mode == "rewrite":
            task = (
                "Perform a lexical and syntactic rewrite of the descriptions in [Step-1] through [Step-3]. "
                "Preserve every forensic target, the step order, and the level of detail. Change wording, "
                "sentence structure, or phrasing only; do not add, remove, merge, or reprioritize evidence."
            )
        else:
            task = (
                "Modify the reasoning expression inside [Step-1] through [Step-3]. You may change emphasis, "
                "reprioritize cues, or substitute relevant forensic angles within the same three categories: "
                "visual, low-level/statistical, and semantic/physics. Keep each category in its original step."
            )

        instruction = """Create ONE child instruction using mode: {mode}.

TASK
{task}

Requirements:
- Begin with this exact first line: {task_line}
- Preserve exactly one occurrence of each heading, in this order: [Step-1], [Step-2], [Step-3], [Step-4].
- Keep this final step verbatim: {final_step}
- Do not mention datasets, generator names, class priors, accuracy, or a preferred verdict.
- Do not state or imply the correct label before inspecting the image.
- Output only the child instruction, without commentary or code fences.
- Keep it below 250 words.

PARENT INSTRUCTION
------------------
{parent}
------------------""".format(
            mode=mode,
            task=task,
            task_line=TASK_LINE,
            final_step=FINAL_STEP,
            parent=parent_prompt,
        )
        return self._call(instruction), mode


def build_initial_pool(generator, pool_size):
    pool = []
    attempts = 0
    while len(pool) < pool_size:
        attempts += 1
        candidate = generator.initialize(BASE_PROMPT)
        if candidate not in pool:
            pool.append(candidate)
        if attempts >= pool_size * 5 and len(pool) < pool_size:
            raise RuntimeError("Could not generate {} distinct initial prompts.".format(pool_size))
    return pool


def extract_verdict(answer):
    step4_matches = list(
        re.finditer(
            r"(?is)(?:\[\s*step\s*[-_]?\s*4\s*\]|step\s*[-_]?\s*4|final\s+assessment)",
            answer,
        )
    )
    segment = answer[step4_matches[-1].start() :] if step4_matches else answer
    verdicts = re.findall(r"(?i)\b(ai[-\s]?generated|generated|authentic|fake|real)\b", segment)
    if not verdicts:
        return None
    return normalize_label(verdicts[0])


class PromptEvaluator:
    def __init__(self, model_name, adapter_path, max_new_tokens):
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from tqdm import tqdm
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "The VLLM runtime is incomplete. Use the same Python 3.11+ environment as UC-VLM "
                "with current transformers, peft, qwen-vl-utils, torch, and tqdm installed."
            ) from exc

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.tqdm = tqdm
        print("[MODEL] Loading processor: {}".format(model_name))
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("[MODEL] Loading base VLLM: {}".format(model_name))
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required when --adapter-path is provided.") from exc
            print("[MODEL] Loading Stage-1 visual adapter: {}".format(adapter_path))
            base_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            print("[MODEL] No adapter supplied; treating --model-name as the merged Stage-1 model.")
        self.model = base_model
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.cache = {}

    def predict(self, image_path, prompt):
        cache_key = (image_path, prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            self.cache[cache_key] = None
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
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        trimmed_ids = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated_ids)]
        answer = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        prediction = extract_verdict(answer)
        self.cache[cache_key] = prediction
        return prediction

    def accuracy(self, prompt, examples):
        correct = 0
        for image_path, label in self.tqdm(examples, desc="Evaluating", leave=False):
            correct += self.predict(image_path, prompt) == label
        return correct / len(examples)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_path.replace(path)


def refill_pool(elites, pool_size, generator, rng):
    next_pool = list(elites)
    mutation_log = []
    attempts = 0
    while len(next_pool) < pool_size:
        attempts += 1
        parent = rng.choice(elites)
        child, mode = generator.mutate(parent)
        if child not in next_pool:
            next_pool.append(child)
            mutation_log.append({"mode": mode, "parent": parent, "child": child})
        if attempts >= pool_size * 8 and len(next_pool) < pool_size:
            raise RuntimeError("Could not refill the pool with distinct child prompts.")
    return next_pool, mutation_log


def main():
    args = parse_args()
    try:
        validate_args(args)
    except (ValueError, FileNotFoundError, EnvironmentError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    random.seed(args.seed)
    try:
        import torch
    except ImportError as exc:
        print("error: PyTorch is required to run prompt evolution.", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = EvolutionConfig(
        model_name=args.model_name,
        adapter_path=str(args.adapter_path.resolve()) if args.adapter_path is not None else None,
        dataset_json=str(args.dataset_json.resolve()),
        output_dir=str(args.output_dir.resolve()),
        openai_model=args.openai_model,
        pool_size=args.pool_size,
        top_k=args.top_k,
        rounds=args.rounds,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        openai_max_tokens=args.openai_max_tokens,
        openai_temperature=args.openai_temperature,
    )
    write_json(args.output_dir / "config.json", asdict(config))

    examples = load_balanced_validation_set(args.dataset_json, args.samples_per_class, rng)
    generator = PromptGenerator(
        args.openai_model, args.openai_temperature, args.openai_max_tokens, rng
    )
    print("[EVO] Generating {} initial candidate prompts with {}.".format(args.pool_size, args.openai_model))
    prompt_pool = build_initial_pool(generator, args.pool_size)
    evaluator = PromptEvaluator(args.model_name, args.adapter_path, args.max_new_tokens)

    print("[BASELINE] Evaluating the non-reasoning classification prompt.")
    baseline_accuracy = evaluator.accuracy(BASIC_PROMPT, examples)
    baseline_result = {
        "prompt": BASIC_PROMPT,
        "accuracy": baseline_accuracy,
        "round": "baseline",
    }
    print("[BASELINE] Accuracy: {:.2%}".format(baseline_accuracy))
    write_json(args.output_dir / "baseline.json", baseline_result)

    # The basic prompt is the incumbent. An evolved prompt must strictly beat it.
    global_best = dict(baseline_result)
    write_json(args.output_dir / "best_prompt.json", global_best)
    history = []

    for round_index in range(args.rounds):
        print("\n=== Evolution round {}/{} ===".format(round_index + 1, args.rounds))
        scored_pool = []
        for prompt_index, prompt in enumerate(prompt_pool):
            accuracy = evaluator.accuracy(prompt, examples)
            print("Prompt {}/{} accuracy: {:.2%}".format(prompt_index + 1, len(prompt_pool), accuracy))
            scored_pool.append({"prompt": prompt, "accuracy": accuracy})

        scored_pool.sort(key=lambda item: item["accuracy"], reverse=True)
        elites = [item["prompt"] for item in scored_pool[: args.top_k]]
        best_this_round = scored_pool[0]
        if best_this_round["accuracy"] > global_best["accuracy"]:
            global_best = {
                "prompt": best_this_round["prompt"],
                "accuracy": best_this_round["accuracy"],
                "round": round_index,
            }
            write_json(args.output_dir / "best_prompt.json", global_best)

        round_payload = {
            "round": round_index,
            "scores": scored_pool,
            "elite_prompts": elites,
            "best_so_far": global_best,
            "mutations": [],
        }

        if round_index + 1 < args.rounds:
            prompt_pool, mutation_log = refill_pool(elites, args.pool_size, generator, rng)
            round_payload["mutations"] = mutation_log

        history.append(round_payload)
        write_json(args.output_dir / "round_{:03d}.json".format(round_index), round_payload)

    final_result = {
        "config": asdict(config),
        "baseline": baseline_result,
        "best": global_best,
        "history": history,
    }
    write_json(args.output_dir / "final_evolution_result.json", final_result)
    print("\n[EVO] Finished. Best accuracy: {:.2%}".format(global_best["accuracy"]))
    print("[EVO] Best prompt saved to: {}".format(args.output_dir / "best_prompt.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
