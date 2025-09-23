#!/usr/bin/env python3
"""
Evaluate a Whisper model (local path or HF Hub repo) and compute WER/CER.

Examples:
  # Evaluate local model on Common Voice Greek test split
  python process_evailuate.py --model ./models/whisper-greek --dataset mozilla-foundation/common_voice_13_0 --config el --split test[:200]

  # Evaluate HF Hub model on FLEURS
  python process_evailuate.py --model your-username/greek-whisper --dataset google/fleurs --config el --split test --ref_key transcription
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset, Audio
import evaluate
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper model (WER/CER)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--model", required=True, help="Model path or HF repo id")
    parser.add_argument("--dataset", required=True, help="HF dataset name (e.g., mozilla-foundation/common_voice_13_0)")
    parser.add_argument("--config", default=None, help="Dataset config (e.g., el)")
    parser.add_argument("--split", default="test", help="Dataset split (e.g., test, test[:200])")
    parser.add_argument("--ref_key", default="text", help="Reference text column (default: text)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference (default: 4)")
    parser.add_argument("--max_new_tokens", type=int, default=225, help="Max new tokens for generation")
    parser.add_argument("--device", default=None, help="cuda|cpu (default: auto)")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    ds_kwargs = {}
    if args.config:
        ds_kwargs["name"] = args.config
    print(f"Loading dataset: {args.dataset}, config={args.config}, split={args.split}")
    try:
        dataset = load_dataset(args.dataset, split=args.split, **ds_kwargs)
    except ValueError as e:
        # Fallback: retry without config if provided config doesn't exist
        if args.config is not None and "BuilderConfig" in str(e):
            print(f"Config '{args.config}' not found for {args.dataset}. Retrying without config...")
            dataset = load_dataset(args.dataset, split=args.split)
        else:
            raise
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Load model and processor
    print(f"Loading model: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # Batched inference
    preds = []
    refs = []

    def iter_batches(iterable, n):
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch

    with torch.inference_mode():
        for batch in iter_batches(dataset, args.batch_size):
            audios = [ex["audio"]["array"] for ex in batch]
            srs = [ex["audio"]["sampling_rate"] for ex in batch]
            inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated_ids = model.generate(inputs["input_features"], max_new_tokens=args.max_new_tokens)
            batch_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            preds.extend([p.strip() for p in batch_preds])
            refs.extend([ex.get(args.ref_key, "").strip() for ex in batch])

    wer = 100 * wer_metric.compute(predictions=preds, references=refs)
    cer = 100 * cer_metric.compute(predictions=preds, references=refs)

    print(f"WER: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
