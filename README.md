# UC-VLM

Consistency-driven learning for AI-generated image detection with vision-language large models.

UC-VLM adapts a general-purpose VLLM to distinguish authentic photographs from AI-generated images using only binary authenticity labels. The same `authentic/generated` supervision is reused across three coordinated stages:

1. **Visual Adaptation** strengthens sensitivity to local, non-semantic forensic cues.
2. **Prompt Evolution** searches for an effective and stable authenticity instruction.
3. **Label-Conditioned Regeneration** reconstructs language-side training targets without human-written rationales.

The implementation is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and uses Qwen2.5-VL-7B-Instruct as the default backbone.

> **Important:** Generated reasoning is used as an auxiliary language-side training signal. It is not a verified forensic explanation and may contain hallucinations.

## Repository Layout

```text
UC-VLM/
├── README.md
├── build_visual_dataset.py
├── merge_visual_lora.py
├── prompt_evolution.py
├── build_evolved_prompt_dataset.py
├── build_label_conditioned_dataset.py
├── run_stage3_language_training.sh
└── UC-VLM/                              # Modified LLaMA-Factory source tree
    ├── data/
    │   ├── dataset_info.json
    │   ├── sdv4_vt.json         # Generated locally; not recommended for Git
    │   └── sdv4_label.json
    ├── examples/train_lora/
    │   ├── qwen2_5_vl_visual_adaptation.yaml
    │   └── qwen2_5_vl_label_conditioned_sft.yaml
    ├── output/
    │   └── qwen2_5vl_visual_merged/
    ├── saves/uc_vlm/
    └── src/llamafactory/
        ├── data/mm_plugin.py
        ├── hparams/model_args.py
        └── model/patcher.py
```

Generated datasets, model weights, adapters, caches, and API credentials should not be committed to a public repository.

### Python Environment

Create and activate an isolated environment:

```bash
conda create -n uc-vlm python=3.11 -y
conda activate uc-vlm
```

Install the modified LLaMA-Factory source:

```bash
cd /path/to/UC-VLM/UC-VLM
pip install -e .
pip install -r requirements/metrics.txt
```

Install the additional pipeline dependencies if they are not already present:

```bash
pip install openai peft Pillow qwen-vl-utils tqdm
```

Verify the environment:

```bash
python --version
llamafactory-cli version
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('Qwen2.5-VL available')"
```

## Quick Start

Define the repository locations once:

```bash
export UC_VLM_ROOT=/path/to/UC-VLM
export LF_ROOT="${UC_VLM_ROOT}/UC-VLM"
```

## 1. Build the Binary Visual-Adaptation Dataset

Input images are expected in separate authentic and generated directories. Nested folders are supported.

```bash
python "${UC_VLM_ROOT}/build_visual_dataset.py" \
  --real-dir /path/to/train/real \
  --generated-dir /path/to/train/fake \
  --output "${LF_ROOT}/data/sdv4_vt.json" \
  --limit-per-class 18000 \
  --seed 42
```

## 2. Stage 1: Visual Adaptation

Start training from the nested LLaMA-Factory directory:

```bash
cd "${LF_ROOT}"
llamafactory-cli train examples/train_lora/qwen2_5_vl_visual_adaptation.yaml
```

Stage-1 adapters are written by default to:

```text
UC-VLM/saves/uc_vlm/qwen2_5vl_visual_adaptation/lora/sft/
```

## 3. Merge the Stage-1 Visual LoRA

Merge the visual adapter into the backbone before prompt evolution and language-side training:

```bash
python "${UC_VLM_ROOT}/merge_visual_lora.py" \
  --base-model Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter-path "${LF_ROOT}/saves/uc_vlm/qwen2_5vl_visual_adaptation/lora/sft" \
  --output-dir "${LF_ROOT}/output/qwen2_5vl_visual_merged" \
  --dtype bfloat16
```

## 4. Stage 2: Prompt Evolution

Set the API key through the environment. Never store it in source files:

```bash
export OPENAI_API_KEY="your_api_key"
```

Run prompt evolution with the merged visual model:

```bash
python "${UC_VLM_ROOT}/prompt_evolution.py" \
  --model-name "${LF_ROOT}/output/qwen2_5vl_visual_merged" \
  --dataset-json "${LF_ROOT}/data/only_sdv4_vt_11.json" \
  --output-dir "${UC_VLM_ROOT}/prompt_evolution_outputs" \
  --openai-model gpt-4o \
  --pool-size 5 \
  --top-k 3 \
  --rounds 15 \
  --samples-per-class 1000 \
  --seed 2025
```

Alternatively, evaluate the unmerged backbone plus adapter:

```bash
python "${UC_VLM_ROOT}/prompt_evolution.py" \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter-path /path/to/visual_adapter \
  --dataset-json /path/to/validation.json
```

## 5. Build the Label-Conditioned Regeneration Dataset

This stage reconstructs assistant content rather than merely replacing the user prompt.

### Option A: Use the Evolved Prompt

```bash
python "${UC_VLM_ROOT}/build_label_conditioned_dataset.py" \
  --model-name "${LF_ROOT}/output/qwen2_5vl_visual_merged" \
  --source-dataset "${LF_ROOT}/data/only_sdv4_vt_11.json" \
  --prompt-result "${UC_VLM_ROOT}/prompt_evolution_outputs/best_prompt.json" \
  --output "${LF_ROOT}/data/only_sdv4_label_conditioned.json" \
  --limit-per-class 3000 \
  --seed 42
```

### Option B: Supply a Manual Prompt File

```bash
python "${UC_VLM_ROOT}/build_label_conditioned_dataset.py" \
  --model-name "${LF_ROOT}/output/qwen2_5vl_visual_merged" \
  --source-dataset "${LF_ROOT}/data/only_sdv4_vt_11.json" \
  --prompt-file /path/to/manual_prompt.txt \
  --output "${LF_ROOT}/data/only_sdv4_label_conditioned.json"
```

### Option C: Paste a Prompt Interactively

Omit all prompt-source options:

```bash
python "${UC_VLM_ROOT}/build_label_conditioned_dataset.py" \
  --model-name "${LF_ROOT}/output/qwen2_5vl_visual_merged"
```

Paste the prompt and enter a line containing only `END` when finished.

The dataset is registered as `uc_vlm_label_conditioned`.

## 6. Stage 3: Language-Side SFT

The final stage freezes the adapted visual pathway and trains only language-side LoRA parameters:

```bash
cd "${LF_ROOT}"
llamafactory-cli train examples/train_lora/qwen2_5_vl_label_conditioned_sft.yaml
```

From the outer repository root, the preflight launcher can also be used:

```bash
cd "${UC_VLM_ROOT}"
./run_stage3_language_training.sh
```

The launcher derives the repository path automatically. Set `UC_VLM_LF_ROOT` only when the nested LLaMA-Factory directory is stored elsewhere.

The default language adapter output is:

```text
UC-VLM/saves/uc_vlm/qwen2_5vl_language_regeneration/lora/sft/
```
