# Fine-Tuning Module

This directory contains scripts to fine-tune models for the Movie Scene Analyzer, addressing the professor's feedback about fine-tuning existing LLMs.

## Overview

| Script | Purpose | Output |
|--------|---------|--------|
| `create_training_data.py` | Generate training data from your analyses | `training_data/*.jsonl` |
| `openai_finetune.py` | Fine-tune GPT-4o-mini on scene analysis | Fine-tuned model ID |
| `emotion_finetune.py` | Fine-tune wav2vec2 on movie emotions | `emotion_model/` |
| `use_finetuned.py` | Use fine-tuned models in your pipeline | — |

## Quick Start

### 1. Fine-Tune GPT-4o-mini (Scene Analysis)

```bash
# Step 1: Generate training data from your existing analyses
cd /path/to/honoursProject
python -m finetune.create_training_data --format openai --data-dir .

# Step 2: Start fine-tuning job
export OPENAI_API_KEY='your-key'
python -m finetune.openai_finetune --train finetune/training_data/train.jsonl

# Step 3: Wait for completion (or check status)
python -m finetune.openai_finetune --wait <job_id>

# Step 4: Test the fine-tuned model
python -m finetune.openai_finetune --test <model_id>
```

**Cost estimate:** ~$0.80 for 100 training examples

### 2. Fine-Tune Emotion Classifier

```bash
# Option A: Fine-tune on MELD dataset (TV show dialogues)
pip install datasets peft accelerate
python -m finetune.emotion_finetune --dataset meld --epochs 3

# Option B: Fine-tune on your own labeled clips
# First, extract audio segments from your video
python -m finetune.emotion_finetune --extract-segments

# Then train on them
python -m finetune.emotion_finetune --custom-data finetune/emotion_data
```

### 3. Use Fine-Tuned Models

```python
from finetune import (
    print_model_status,
    analyze_scene_finetuned,
    classify_emotion_finetuned
)

# Check what's available
print_model_status()

# Use fine-tuned scene analysis
result = analyze_scene_finetuned(keyframes, conflict_summary)

# Use fine-tuned emotion classification
emotion, confidence, all_scores = classify_emotion_finetuned("audio.wav")
```

## Training Data Format

### OpenAI Format (JSONL)
```json
{"messages": [
  {"role": "system", "content": "You are an expert film analyst..."},
  {"role": "user", "content": "Analyze this scene..."},
  {"role": "assistant", "content": "## Scene Analysis\n..."}
]}
```

### Alpaca Format (for local LLMs)
```json
{
  "instruction": "Analyze the following movie scene...",
  "input": "[transcript + emotions]",
  "output": "[analysis]"
}
```

## Fine-Tuning Options Comparison

| Approach | Cost | Effort | GPU Needed | Best For |
|----------|------|--------|------------|----------|
| OpenAI GPT-4o-mini | ~$0.80/100 examples | Low | No | Quick results |
| Emotion classifier (LoRA) | Free | Medium | Yes (Colab OK) | Better emotion accuracy |
| Local LLM (Llama/Mistral) | Free | High | Yes | Full control |

## Using with the Main Pipeline

After fine-tuning, you can run the pipeline with your fine-tuned models:

```bash
# The pipeline will auto-detect fine-tuned models
python pipeline.py --use-finetuned

# Or specify explicitly
python pipeline.py --gpt-model ft:gpt-4o-mini-xxx --emotion-model ./finetune/emotion_model
```

## Tips

1. **More training data = better results**: Analyze multiple video clips to generate more training examples
2. **Quality over quantity**: Review your best GPT-4o + Claude cross-validated outputs
3. **LoRA for efficiency**: Use `--use-lora` flag for emotion fine-tuning to reduce memory requirements
4. **Evaluate**: Compare fine-tuned vs base model on held-out examples

## Files

```
finetune/
├── __init__.py              # Module exports
├── README.md                # This file
├── create_training_data.py  # Generate training data
├── openai_finetune.py       # OpenAI fine-tuning
├── emotion_finetune.py      # Emotion model fine-tuning
├── use_finetuned.py         # Inference with fine-tuned models
├── training_data/           # Generated training data (created by create_training_data.py)
│   ├── train.jsonl
│   └── validation.jsonl
├── emotion_model/           # Fine-tuned emotion model (created by emotion_finetune.py)
│   ├── config.json
│   ├── model.safetensors
│   └── training_info.json
└── model_info.json          # Fine-tuned GPT model info (created by openai_finetune.py)
```
