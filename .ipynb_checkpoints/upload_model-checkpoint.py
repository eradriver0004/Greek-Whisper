#!/usr/bin/env python3
"""
Upload an existing trained model to Hugging Face Hub.
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Upload existing Whisper model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload model with default settings
  python upload_model.py --model_path ./models/whisper-greek

  # Upload with custom repo name and description
  python upload_model.py --model_path ./models/whisper-greek --repo_name my-greek-whisper --description "Greek Whisper model trained on Common Voice"

  # Upload with specific language code
  python upload_model.py --model_path ./models/whisper-greek --language_code el --repo_name greek-whisper-tiny
        """
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )

    # Optional arguments
    parser.add_argument(
        "--repo_name",
        type=str,
        help="Repository name for Hugging Face Hub (default: uses model directory name)"
    )
    parser.add_argument(
        "--language_code",
        type=str,
        default="el",
        help="Language code for the model (default: el)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Fine-tuned Whisper model for Greek speech recognition",
        help="Model description"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Add fine-tuned Greek Whisper model",
        help="Commit message for the upload"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Check if model files exist
    required_files = ["config.json", "pytorch_model.bin"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(args.model_path, f))]
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Make sure this is a valid Hugging Face model directory")
        sys.exit(1)

    # Set repo name if not provided
    if not args.repo_name:
        args.repo_name = os.path.basename(os.path.abspath(args.model_path))

    print("=" * 60)
    print("UPLOADING MODEL TO HUGGING FACE HUB")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Repository: {args.repo_name}")
    print(f"Language code: {args.language_code}")
    print(f"Description: {args.description}")
    print(f"Private: {args.private}")
    print("=" * 60)

    try:
        # Import required libraries
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from huggingface_hub import HfApi, HfFolder
        from config import HF_API_KEY

        # Login to Hugging Face
        print("Logging in to Hugging Face...")
        HfFolder.save_token(HF_API_KEY)
        api = HfApi()

        # Load the model and processor
        print("Loading model and processor...")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
        processor = WhisperProcessor.from_pretrained(args.model_path)

        # Set correct language code
        if hasattr(model.config, 'language'):
            model.config.language = args.language_code
            print(f"Set language code to: {args.language_code}")

        # Create repository if it doesn't exist
        print(f"Creating repository: {args.repo_name}")
        try:
            api.create_repo(
                repo_id=args.repo_name,
                repo_type="model",
                private=args.private,
                exist_ok=True
            )
            print("✓ Repository created/verified")
        except Exception as e:
            print(f"⚠ Repository creation warning: {e}")

        # Upload model
        print("Uploading model...")
        model.push_to_hub(
            args.repo_name,
            commit_message=args.commit_message
        )
        print("✓ Model uploaded successfully")

        # Upload processor
        print("Uploading processor...")
        processor.push_to_hub(
            args.repo_name,
            commit_message=args.commit_message
        )
        print("✓ Processor uploaded successfully")

        # Create model card
        print("Creating model card...")
        model_card_content = f"""---
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

# {args.repo_name}

{args.description}

## Model Details

- **Language**: Greek (el)
- **Model Type**: Whisper (fine-tuned)
- **Base Model**: {model.config.architectures[0] if hasattr(model.config, 'architectures') else 'WhisperForConditionalGeneration'}

## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load model and processor
processor = WhisperProcessor.from_pretrained("{args.repo_name}")
model = WhisperForConditionalGeneration.from_pretrained("{args.repo_name}")

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

        # Upload model card
        with open(os.path.join(args.model_path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)

        api.upload_file(
            path_or_fileobj=os.path.join(args.model_path, "README.md"),
            path_in_repo="README.md",
            repo_id=args.repo_name,
            repo_type="model",
            commit_message="Add model card"
        )
        print("✓ Model card uploaded successfully")

        print("=" * 60)
        print("UPLOAD COMPLETED SUCCESSFULLY!")
        print(f"Model available at: https://huggingface.co/{args.repo_name}")
        print("=" * 60)

    except ImportError as e:
        print(f"Error: Missing required library: {e}")
        print("Please install: pip install transformers huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Error uploading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
