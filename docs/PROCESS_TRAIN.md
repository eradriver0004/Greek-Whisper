# Process Train Documentation

## Overview

`process_train.py` is a command-line script for training Whisper models on Greek speech recognition tasks. It provides a user-friendly interface for configuring and running training sessions with various datasets and model configurations.

## Prerequisites

- Python 3.9-3.11
- PyTorch with CUDA support (recommended)
- Hugging Face account and API token
- Required dependencies (see main README)

## Command Overview

```bash
python process_train.py [OPTIONS]
```

## Required Arguments

None - all arguments have sensible defaults.

## Optional Arguments

### Model Configuration
- `--model_name`: Hugging Face model name or local path (default: `openai/whisper-small`)
- `--existing_model`: Load existing model from model_name path (flag)

### Dataset Configuration
- `--dataset_name`: Dataset name or local path (default: `mozilla-foundation/common_voice_13_0`)
- `--ref_key`: Column name containing transcriptions (default: `sentence`)

### Language Configuration
- `--language`: Language name (default: `Greek`)
- `--language_code`: Language code (default: `el`)

### Output Configuration
- `--output_dir`: Directory to save trained model (default: `./models/whisper-greek`)
- `--save_to_hf`: Upload model to Hugging Face Hub (flag)

### Training Parameters
- `--max_steps`: Maximum training steps (default: `4000`)
- `--learning_rate`: Learning rate (default: `1e-5`)
- `--batch_size`: Training batch size (default: `16`)
- `--eval_steps`: Evaluation interval (default: `1000`)
- `--save_steps`: Checkpoint save interval (default: `1000`)

### Utility
- `--verbose`: Enable verbose output (flag)

## Usage Examples

### Basic Training on Common Voice

```bash
python process_train.py --language Greek --language_code el --output_dir ./models/whisper-greek
```

### Training on Custom Dataset

```bash
python process_train.py \
    --dataset_name ./data/my_greek_dataset \
    --ref_key text \
    --output_dir ./models/whisper-custom
```

### Continue Training from Checkpoint

```bash
python process_train.py \
    --model_name ./models/whisper-greek-checkpoint-1000 \
    --existing_model \
    --output_dir ./models/whisper-continued
```

### Training with Custom Parameters

```bash
python process_train.py \
    --model_name openai/whisper-medium \
    --max_steps 8000 \
    --learning_rate 5e-6 \
    --batch_size 8 \
    --output_dir ./models/whisper-medium-greek
```

### Training and Upload to Hugging Face

```bash
python process_train.py \
    --language Greek \
    --language_code el \
    --output_dir ./models/whisper-greek \
    --save_to_hf
```

### Training on FLEURS Dataset

```bash
python process_train.py \
    --dataset_name google/fleurs \
    --language Greek \
    --language_code el \
    --ref_key transcription \
    --output_dir ./models/whisper-greek-fleurs
```

## Available Models

| Model | Parameters | Multilingual | Recommended Use |
|-------|------------|--------------|-----------------|
| `openai/whisper-tiny` | 39M | ✓ | Quick testing |
| `openai/whisper-base` | 74M | ✓ | Development |
| `openai/whisper-small` | 244M | ✓ | **Recommended** |
| `openai/whisper-medium` | 769M | ✓ | High accuracy |
| `openai/whisper-large-v2` | 1550M | ✓ | Best accuracy |
| `openai/whisper-large-v3` | 1550M | ✓ | Latest version |

## Dataset Options

### Hugging Face Datasets
- `mozilla-foundation/common_voice_13_0` (Common Voice)
- `google/fleurs` (FLEURS)
- `facebook/voxpopuli` (VoxPopuli)

### Local Datasets
- Path to local dataset directory
- Must be in Hugging Face dataset format
- Created using `process_prepair_datasets.py`

## Training Process

1. **Initialization**: Load model, tokenizer, and dataset
2. **Data Preparation**: Convert audio to spectrograms, tokenize text
3. **Training Loop**: Forward pass, loss calculation, backpropagation
4. **Evaluation**: Compute WER on validation set
5. **Checkpointing**: Save model at specified intervals
6. **Completion**: Save final model and optionally upload to HF

## Output Structure

```
output_dir/
├── pytorch_model.bin          # Model weights
├── config.json                # Model configuration
├── tokenizer.json             # Tokenizer files
├── tokenizer_config.json
├── preprocessor_config.json
├── runs/                      # TensorBoard logs
│   └── [timestamp]/
└── checkpoint-*/              # Training checkpoints
    ├── pytorch_model.bin
    ├── config.json
    └── ...
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./models/whisper-greek/runs
```

### Key Metrics
- **Loss**: Should decrease over time
- **WER**: Word Error Rate (lower is better)
- **Learning Rate**: Automatically scheduled
- **Gradient Norm**: Should be stable

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   python process_train.py --batch_size 4 --max_steps 2000
   ```

2. **Dataset Loading Error**
   - Check `--ref_key` matches your dataset column names
   - Verify dataset path exists
   - Ensure proper dataset format

3. **Training Not Converging**
   - Increase learning rate: `--learning_rate 5e-5`
   - Add more training data
   - Check data quality

4. **Slow Training**
   - Use GPU acceleration
   - Increase batch size if memory allows
   - Use smaller model for testing

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

## Integration with Other Scripts

### Create Dataset First
```bash
# Create dataset from ELAN annotations
python process_prepair_datasets.py \
    --input data/annotations.eaf \
    --dataset-url your-username/greek-dataset \
    --output-dir data/greek_dataset

# Train on created dataset
python process_train.py \
    --dataset_name data/greek_dataset \
    --ref_key text \
    --output_dir ./models/whisper-greek
```

### Test Trained Model
```bash
# Transcribe with trained model
python process_transcribe.py \
    --input_file data/test.wav \
    --model-size small \
    --language Greek \
    --output data/transcription.json \
    diarized
```

## Notes

- Training time depends on model size, dataset size, and hardware
- Use `--verbose` flag for detailed error information
- Check GPU memory usage during training
- Save checkpoints regularly for long training runs
- Monitor validation WER to avoid overfitting

## References

- [Hugging Face Whisper Documentation](https://huggingface.co/docs/transformers/model_doc/whisper)
- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Common Voice Dataset](https://commonvoice.mozilla.org/)
- [FLEURS Dataset](https://huggingface.co/datasets/google/fleurs)
