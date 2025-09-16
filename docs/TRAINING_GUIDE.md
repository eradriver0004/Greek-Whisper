# Fine-tuning Whisper for Greek Speech Recognition

## Overview

This guide covers how to fine-tune OpenAI's Whisper model for Greek speech recognition using the Hugging Face Transformers library. The training system supports both pre-trained models and custom datasets created from ELAN annotations.

## Prerequisites

### System Requirements
- **Python**: 3.9-3.11 (tested with 3.10.13)
- **PyTorch**: 1.12.1 or compatible
- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space for models and datasets

### Dependencies
Install required packages:
```bash
pip install torch torchaudio transformers datasets librosa soundfile
pip install pyannote.audio openai-whisper
pip install huggingface_hub evaluate
```

## Configuration

### 1. Update `config.py`
Before training, configure your settings in `config.py`:

```python
# Model configuration
BASE_MODEL = "openai/whisper-large-v3"  # or whisper-small, whisper-medium, etc.
model_dir = "./models"  # Directory to save models

# Hugging Face API key (get from https://huggingface.co/settings/tokens)
HF_API_KEY = "your_huggingface_token_here"
```

### 2. Training Parameters
Modify training constants in `training/trainner.py`:

```python
# Training hyperparameters
TRAIN_BATCH_SIZE = 16          # Batch size for training
EVAL_BATCH_SIZE = 8            # Batch size for evaluation
LEARNING_RATE = 1e-5           # Learning rate
WARMUP_STEPS = 500             # Number of warmup steps
MAX_STEPS = 4000               # Maximum training steps
SAVE_STEPS = 1000              # Save checkpoint every N steps
EVAL_STEPS = 1000              # Evaluate every N steps
LOGGING_STEPS = 25             # Log every N steps
```

## Available Models

| Size | Parameters | Multilingual | Recommended Use |
|:----:|:----------:|:------------:|:---------------:|
| tiny | 39M | ✓ | Quick testing |
| base | 74M | ✓ | Development |
| small | 244M | ✓ | **Recommended** |
| medium | 769M | ✓ | High accuracy |
| large | 1550M | ✓ | Best accuracy |

## Training Methods

### Method 1: Using Pre-built Datasets

Train on Hugging Face datasets (Common Voice, FLEURS, etc.):

```python
from training.trainner import WhisperASR

# Initialize trainer
trainer = WhisperASR(
    model_name="openai/whisper-small",
    dataset_name="mozilla-foundation/common_voice_13_0",
    language="Greek",
    language_code="el",
    output_dir="./models/whisper-greek",
    ref_key="sentence",  # Column name containing transcriptions
    save_to_hf=True
)

# Start training
trainer.train()
```

### Method 2: Using Custom Datasets

Train on your own dataset created from ELAN annotations:

```python
from training.trainner import WhisperASR

# Load your custom dataset
trainer = WhisperASR(
    model_name="openai/whisper-small",
    dataset_name="path/to/your/dataset",  # Local dataset path
    language="Greek",
    language_code="el",
    output_dir="./models/whisper-greek-custom",
    ref_key="text",  # Column name in your dataset
    save_to_hf=False
)

# Start training
trainer.train()
```

### Method 3: Continue Training from Existing Model

Resume training from a previously trained model:

```python
trainer = WhisperASR(
    model_name="./models/whisper-greek-checkpoint-1000",  # Path to checkpoint
    dataset_name="mozilla-foundation/common_voice_13_0",
    language="Greek",
    language_code="el",
    output_dir="./models/whisper-greek-continued",
    existing_model=True,  # Load existing model
    ref_key="sentence"
)

trainer.train()
```

## Dataset Preparation

### Using ELAN Annotations

1. **Create annotations in ELAN** (see `docs/ELAN_BASICS.md`)
2. **Convert to dataset**:
   ```bash
   python process_prepair_datasets.py \
       --input data/annotations.eaf \
       --dataset-url your-username/greek-whisper-dataset \
       --output-dir data/greek_dataset \
       --verbose
   ```
3. **Use in training**:
   ```python
   trainer = WhisperASR(
       dataset_name="data/greek_dataset",  # Local dataset
       ref_key="text"  # Column name from your dataset
   )
   ```

### Dataset Format Requirements

Your dataset must have these columns:
- `audio`: Audio data (Hugging Face Audio feature)
- `text` (or `sentence`): Transcription text
- Optional: `speaker_id`, `duration`, etc.

Example dataset structure:
```python
{
    "audio": [audio_data],
    "text": "Καλημέρα, πώς είστε;",
    "speaker_id": "speaker_01",
    "duration": 2.5
}
```

## Training Process

### 1. Data Preparation
The system automatically:
- Loads audio data at 16kHz sampling rate
- Converts audio to log-Mel spectrograms
- Tokenizes text transcriptions
- Creates train/test splits

### 2. Model Initialization
- Loads pre-trained Whisper model
- Sets up feature extractor and tokenizer
- Configures data collator for batching

### 3. Training Loop
- **Forward pass**: Audio → Spectrograms → Model → Predictions
- **Loss calculation**: Cross-entropy between predictions and ground truth
- **Backpropagation**: Update model parameters
- **Evaluation**: Compute WER (Word Error Rate) on validation set

### 4. Checkpointing
- Saves model every `SAVE_STEPS` (default: 1000)
- Saves best model based on validation WER
- Logs training metrics to TensorBoard

## Monitoring Training

### TensorBoard
View training progress:
```bash
tensorboard --logdir ./models/whisper-greek/runs
```

### Key Metrics
- **Loss**: Should decrease over time
- **WER**: Word Error Rate (lower is better)
- **Learning Rate**: Automatically scheduled
- **Gradient Norm**: Should be stable

### Early Stopping
Training stops when:
- `MAX_STEPS` reached
- No improvement in validation WER for several evaluations

## Command Line Training

Create a training script `train_whisper.py`:

```python
#!/usr/bin/env python3
import argparse
from training.trainner import WhisperASR

def main():
    parser = argparse.ArgumentParser(description="Train Whisper for Greek ASR")
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--dataset_name", default="mozilla-foundation/common_voice_13_0")
    parser.add_argument("--language", default="Greek")
    parser.add_argument("--language_code", default="el")
    parser.add_argument("--output_dir", default="./models/whisper-greek")
    parser.add_argument("--ref_key", default="sentence")
    parser.add_argument("--save_to_hf", action="store_true")
    parser.add_argument("--existing_model", action="store_true")
    
    args = parser.parse_args()
    
    trainer = WhisperASR(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        language=args.language,
        language_code=args.language_code,
        output_dir=args.output_dir,
        ref_key=args.ref_key,
        save_to_hf=args.save_to_hf,
        existing_model=args.existing_model
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
```

Run training:
```bash
python train_whisper.py \
    --model_name openai/whisper-small \
    --dataset_name mozilla-foundation/common_voice_13_0 \
    --language Greek \
    --language_code el \
    --output_dir ./models/whisper-greek \
    --ref_key sentence \
    --save_to_hf
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `TRAIN_BATCH_SIZE` and `EVAL_BATCH_SIZE`
   - Enable gradient checkpointing (already enabled)
   - Use smaller model (whisper-tiny or whisper-base)

2. **Dataset Loading Errors**
   - Check `ref_key` matches your dataset column names
   - Verify audio files are accessible
   - Ensure proper dataset format

3. **Training Not Converging**
   - Increase `LEARNING_RATE` (try 5e-5)
   - Add more training data
   - Check data quality and transcriptions

4. **Slow Training**
   - Use GPU acceleration
   - Increase batch size if memory allows
   - Use mixed precision training (fp16)

### Performance Tips

1. **Data Quality**
   - Ensure clean, high-quality audio
   - Verify transcription accuracy
   - Balance speaker diversity

2. **Training Strategy**
   - Start with small model for testing
   - Use validation set for early stopping
   - Monitor WER trends

3. **Resource Management**
   - Use gradient accumulation for larger effective batch sizes
   - Enable mixed precision training
   - Monitor GPU memory usage

## Evaluation

After training, evaluate your model:

```python
from training.trainner import WhisperASR

# Load trained model
trainer = WhisperASR(
    model_name="./models/whisper-greek",
    existing_model=True
)

# Evaluate on test set
results = trainer.evaluate()
print(f"Test WER: {results['wer']:.2f}%")
```

## Saving and Sharing

### Local Saving
Models are automatically saved to `output_dir` with:
- Model weights (`pytorch_model.bin`)
- Configuration (`config.json`)
- Tokenizer files
- Training logs

### Hugging Face Hub
To share your model:
```python
trainer = WhisperASR(
    # ... other parameters ...
    save_to_hf=True  # Enable Hub upload
)
```

Your model will be available at:
`https://huggingface.co/your-username/model-name`

## Next Steps

1. **Test your model** with `process_transcribe.py`
2. **Create more datasets** using ELAN annotations
3. **Fine-tune further** on domain-specific data
4. **Deploy** for production use

## References

- [Hugging Face Whisper Documentation](https://huggingface.co/docs/transformers/model_doc/whisper)
- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Common Voice Dataset](https://commonvoice.mozilla.org/)
- [FLEURS Dataset](https://huggingface.co/datasets/google/fleurs)
