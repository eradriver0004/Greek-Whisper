# Process Evaluate Documentation

Evaluate a Whisper model (local or Hugging Face Hub) and compute WER/CER.

## Overview
`process_evailuate.py` loads a Whisper model (local directory or HF repo), runs batched inference on a Hugging Face dataset split, and reports Word Error Rate (WER) and Character Error Rate (CER).

## Requirements
- Python 3.9+
- Packages:
  - transformers, datasets, evaluate, torch

Install missing packages:
```bash
pip install transformers datasets evaluate torch
```

## Command
```bash
python process_evailuate.py --model MODEL --dataset DATASET [--config CONFIG] [--split SPLIT]
                            [--ref_key REF_KEY] [--batch_size N] [--max_new_tokens N]
                            [--device DEVICE]
```

## Arguments
- `--model` (required): Model path or HF Hub repo id.
  - Examples: `./models/whisper-greek`, `your-username/greek-whisper`
- `--dataset` (required): HF dataset name.
  - Examples: `mozilla-foundation/common_voice_13_0`, `google/fleurs`
- `--config` (optional): Dataset configuration (e.g., language code).
  - Examples: `el`, `el_gr`
- `--split` (optional): Dataset split to evaluate.
  - Default: `test`; Examples: `test[:200]`, `validation`
- `--ref_key` (optional): Reference text column in the dataset.
  - Default: `text`; For FLEURS use `transcription`
- `--batch_size` (optional): Inference batch size. Default: `4`
- `--max_new_tokens` (optional): Max tokens to generate. Default: `225`
- `--device` (optional): `cuda` or `cpu`. Default: auto-detect

## Examples

### 1) Evaluate local model on Common Voice (Greek, small slice)
```bash
python process_evailuate.py \
  --model ./models/whisper-greek \
  --dataset mozilla-foundation/common_voice_13_0 \
  --config el \
  --split test[:200]
```

### 2) Evaluate HF Hub model on FLEURS (full test)
```bash
python process_evailuate.py \
  --model your-username/greek-whisper \
  --dataset google/fleurs \
  --config el \
  --split test \
  --ref_key transcription
```

### 3) Force CPU and smaller batches
```bash
python process_evailuate.py \
  --model ./models/whisper-greek \
  --dataset mozilla-foundation/common_voice_13_0 \
  --config el \
  --split test[:50] \
  --batch_size 1 \
  --device cpu
```

## Output
- Prints final metrics to stdout:
```
WER: 18.42%
CER: 9.77%
```

## Tips
- Start with a small split (e.g., `test[:200]`) for quick estimates, then scale up.
- Ensure audio is auto-cast to `Audio(16000)` (handled by the script).
- For FLEURS or custom datasets, set `--ref_key` to the correct text column.
- Use `cuda` if available for faster inference.
