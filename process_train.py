#!/usr/bin/env python3
"""
Training script for Whisper Greek ASR model.

This script provides a command-line interface for training Whisper models
on Greek speech recognition tasks using various datasets.
"""

import argparse
import os
import sys
from pathlib import Path

# Add training directory to path
current_dir = Path(__file__).parent
training_dir = current_dir / "training"
sys.path.insert(0, str(training_dir))

from training.trainner import WhisperASR


def main():
    parser = argparse.ArgumentParser(
        description="Train Whisper for Greek Speech Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on Common Voice dataset
  python process_train.py --language Greek --language_code el --output_dir ./models/whisper-greek

  # Train on custom dataset
  python process_train.py --dataset_name ./data/my_dataset --ref_key text --output_dir ./models/whisper-custom

  # Continue training from checkpoint
  python process_train.py --model_name ./models/whisper-greek-checkpoint-1000 --existing_model --output_dir ./models/whisper-continued

  # Train and upload to Hugging Face
  python process_train.py --language Greek --language_code el --output_dir ./models/whisper-greek --save_to_hf

  # Train with custom parameters
  python process_train.py --model_name openai/whisper-medium --max_steps 8000 --learning_rate 5e-6 --batch_size 8
        """
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-tiny",
        help="Hugging Face model name or path to local model (default: openai/whisper-tiny)"
    )
    parser.add_argument(
        "--existing_model",
        action="store_true",
        help="Load existing model from model_name path"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mozilla-foundation/common_voice_13_0",
        help="Dataset name from Hugging Face or path to local dataset (default: mozilla-foundation/common_voice_13_0)"
    )
    parser.add_argument(
        "--ref_key",
        type=str,
        default="text",
        help="Column name containing transcriptions in dataset (default: text)"
    )

    # Language parameters
    parser.add_argument(
        "--language",
        type=str,
        default="Greek",
        help="Language name for training (default: Greek)"
    )
    parser.add_argument(
        "--language_code",
        type=str,
        default="el",
        help="Language code for dataset configuration (default: el)"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/whisper-greek",
        help="Directory to save trained model (default: ./models/whisper-greek)"
    )
    parser.add_argument(
        "--save_to_hf",
        action="store_true",
        help="Upload model to Hugging Face Hub"
    )

    # Training parameters
    parser.add_argument(
        "--max_steps",
        type=int,
        default=4000,
        help="Maximum training steps (default: 4000)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluation steps interval (default: 1000)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint steps interval (default: 1000)"
    )

    # Utility parameters
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.existing_model and not os.path.exists(args.model_name):
        print(f"Error: Model path does not exist: {args.model_name}")
        sys.exit(1)

    if not os.path.exists(args.dataset_name) and not args.dataset_name.startswith(("mozilla-foundation", "google", "facebook")):
        print(f"Warning: Dataset path may not exist: {args.dataset_name}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("WHISPER GREEK ASR TRAINING")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Language: {args.language} ({args.language_code})")
    print(f"Reference key: {args.ref_key}")
    print(f"Output directory: {args.output_dir}")
    print(f"Existing model: {args.existing_model}")
    print(f"Save to HF: {args.save_to_hf}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Eval steps: {args.eval_steps}")
    print(f"Save steps: {args.save_steps}")
    print("=" * 60)

    try:
        # Initialize trainer
        print("Initializing trainer...")
        trainer = WhisperASR(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            existing_model=args.existing_model,
            language=args.language,
            language_code=args.language_code,
            save_to_hf=args.save_to_hf,
            output_dir=args.output_dir,
            ref_key=args.ref_key
        )

        # Update training parameters if specified
        if args.max_steps != 4000 or args.learning_rate != 1e-5 or args.batch_size != 16:
            import training.trainner as trainner_module
            
            if args.max_steps != 4000:
                trainner_module.MAX_STEPS = args.max_steps
                print(f"Updated MAX_STEPS to {args.max_steps}")
            
            if args.learning_rate != 1e-5:
                trainner_module.LEARNING_RATE = args.learning_rate
                print(f"Updated LEARNING_RATE to {args.learning_rate}")
            
            if args.batch_size != 16:
                trainner_module.TRAIN_BATCH_SIZE = args.batch_size
                print(f"Updated TRAIN_BATCH_SIZE to {args.batch_size}")
            
            if args.eval_steps != 1000:
                trainner_module.EVAL_STEPS = args.eval_steps
                print(f"Updated EVAL_STEPS to {args.eval_steps}")
            
            if args.save_steps != 1000:
                trainner_module.SAVE_STEPS = args.save_steps
                print(f"Updated SAVE_STEPS to {args.save_steps}")

        # Start training
        print("Starting training...")
        trainer.train()

        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Model saved to: {args.output_dir}")
        
        if args.save_to_hf:
            print("Uploading model to Hugging Face Hub...")
            try:
                # Set correct language codes for upload
                trainer.model.config.language = args.language_code
                if hasattr(trainer.model.config, 'language_bcp47'):
                    trainer.model.config.language_bcp47 = f"{args.language_code}-GR"
                
                # Create a proper model card
                repo_name = args.output_dir.split('/')[-1]
                model_card = f"""---
language: {args.language_code}
license: mit
tags:
- whisper
- speech-recognition
- greek
- audio
- automatic-speech-recognition
pipeline_tag: automatic-speech-recognition
---

# {repo_name}

Fine-tuned Whisper model for Greek speech recognition.

## Model Details

- **Language**: Greek ({args.language_code})
- **Model Type**: Whisper (fine-tuned)
- **Base Model**: {trainer.model.config.architectures[0] if hasattr(trainer.model.config, 'architectures') else 'WhisperForConditionalGeneration'}

## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load model and processor
processor = WhisperProcessor.from_pretrained("{repo_name}")
model = WhisperForConditionalGeneration.from_pretrained("{repo_name}")

# Load audio
audio, sampling_rate = librosa.load("audio.wav", sr=16000)

# Process audio
input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(transcription)
```

## Training Details

This model was fine-tuned on Greek speech data using the Hugging Face Transformers library.

## License

MIT License
"""
                
                # Save model card
                with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
                    f.write(model_card)
                
                trainer.model.push_to_hub(
                    repo_name,
                    commit_message=f"Add Greek Whisper model trained on {args.dataset_name}"
                )
                print("✓ Model uploaded to Hugging Face Hub")
            except Exception as e:
                print(f"⚠ Failed to upload to Hugging Face Hub: {e}")
                print("Model is still saved locally")
        
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()