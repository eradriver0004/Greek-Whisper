# Whisper Training Quick Start

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install torch torchaudio transformers datasets librosa soundfile pyannote.audio
   ```

2. **Configure settings** in `config.py`:
   ```python
   HF_API_KEY = "your_huggingface_token"
   BASE_MODEL = "openai/whisper-small"
   ```

3. **Start training**:
   ```python
   from training.trainner import WhisperASR
   
   trainer = WhisperASR(
       model_name="openai/whisper-small",
       dataset_name="mozilla-foundation/common_voice_13_0",
       language="Greek",
       language_code="el",
       output_dir="./models/whisper-greek",
       ref_key="sentence"
   )
   
   trainer.train()
   ```

## Common Commands

### Train on Common Voice
```python
trainer = WhisperASR(
    model_name="openai/whisper-small",
    dataset_name="mozilla-foundation/common_voice_13_0",
    language="Greek",
    language_code="el",
    output_dir="./models/whisper-greek",
    ref_key="sentence"
)
trainer.train()
```

### Train on Custom Dataset
```python
trainer = WhisperASR(
    model_name="openai/whisper-small",
    dataset_name="data/my_greek_dataset",  # Local dataset
    language="Greek",
    language_code="el",
    output_dir="./models/whisper-greek-custom",
    ref_key="text"  # Column name in your dataset
)
trainer.train()
```

### Continue Training
```python
trainer = WhisperASR(
    model_name="./models/whisper-greek-checkpoint-1000",
    dataset_name="mozilla-foundation/common_voice_13_0",
    language="Greek",
    language_code="el",
    output_dir="./models/whisper-greek-continued",
    existing_model=True
)
trainer.train()
```

## Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_name` | Hugging Face model or local path | `"openai/whisper-small"` |
| `dataset_name` | Dataset name or local path | `"mozilla-foundation/common_voice_13_0"` |
| `language` | Language name | `"Greek"` |
| `language_code` | Language code | `"el"` |
| `output_dir` | Save directory | `"./models/whisper-greek"` |
| `ref_key` | Transcription column | `"sentence"` or `"text"` |
| `existing_model` | Load existing model | `True` or `False` |
| `save_to_hf` | Upload to Hugging Face | `True` or `False` |

## Monitoring

- **TensorBoard**: `tensorboard --logdir ./models/whisper-greek/runs`
- **Checkpoints**: Saved every 1000 steps in `output_dir`
- **Logs**: Console output shows loss and WER

## Troubleshooting

- **Out of memory**: Reduce batch size in `trainner.py`
- **Slow training**: Use GPU, increase batch size
- **Poor results**: Check data quality, try larger model
