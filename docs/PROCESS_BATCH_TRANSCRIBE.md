# Batch Audio Transcription Script

## Overview

The `process_batch_transcribe.py` script enables efficient batch processing of multiple audio files for transcription. It supports both basic and diarized transcription with parallel processing capabilities for significantly faster processing.

## Features

- **Parallel Processing**: Process multiple audio files simultaneously using multiprocessing
- **Batch Processing**: Process files in configurable batches to manage memory usage
- **Multiple Formats**: Support for various audio formats (WAV, MP3, M4A, FLAC, OGG, AAC)
- **Two Transcription Types**:
  - **Basic**: Simple transcription using Whisper
  - **Diarized**: Speaker diarization + transcription
- **Flexible Output**: JSON or EAF output formats for diarized transcription
- **Progress Tracking**: Real-time progress monitoring and detailed summaries
- **Error Handling**: Robust error handling with detailed error reporting

## Prerequisites

```bash
# Install required dependencies
pip install torch torchaudio transformers datasets librosa soundfile
pip install pyannote.audio  # For diarized transcription
pip install pydub  # For audio format support
```

## Usage

### Basic Syntax

```bash
python process_batch_transcribe.py <transcription_type> <input_path> <output_dir> [options]
```

### Transcription Types

#### 1. Basic Transcription
```bash
python process_batch_transcribe.py basic /path/to/audio/files /path/to/output
```

#### 2. Diarized Transcription
```bash
python process_batch_transcribe.py diarized /path/to/audio/files /path/to/output
```

## Arguments

### Required Arguments

- **`input_path`**: Path to audio file or directory containing audio files
- **`output_dir`**: Directory to save transcription results

### Optional Arguments

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--model-size` | tiny, base, small, medium, large | base | Whisper model size |
| `--language` | Any language name | Greek | Language for transcription |
| `--batch-size` | Integer | None (process all) | Number of files per batch |
| `--max-workers` | Integer | Auto (CPU count) | Maximum parallel workers |
| `--output-format` | json, eaf | json | Output format (diarized only) |
| `--verbose` | Flag | True | Print progress information |
| `--quiet` | Flag | False | Suppress progress output |

## Examples

### 1. Basic Batch Transcription

```bash
# Process all audio files in a directory
python process_batch_transcribe.py basic ./audio_files ./transcriptions

# Use large model for better accuracy
python process_batch_transcribe.py basic ./audio_files ./transcriptions --model-size large

# Process in smaller batches to manage memory
python process_batch_transcribe.py basic ./audio_files ./transcriptions --batch-size 10
```

### 2. Diarized Batch Transcription

```bash
# Basic diarized transcription
python process_batch_transcribe.py diarized ./audio_files ./transcriptions

# Save as EAF files for ELAN
python process_batch_transcribe.py diarized ./audio_files ./transcriptions --output-format eaf

# Use specific language and model
python process_batch_transcribe.py diarized ./audio_files ./transcriptions --language English --model-size large
```

### 3. Memory-Optimized Processing

```bash
# Limit parallel workers to reduce memory usage
python process_batch_transcribe.py diarized ./audio_files ./transcriptions --max-workers 2 --batch-size 5

# Use smaller model for faster processing
python process_batch_transcribe.py basic ./audio_files ./transcriptions --model-size tiny --max-workers 4
```

### 4. Large Dataset Processing

```bash
# Process thousands of files efficiently
python process_batch_transcribe.py basic ./large_dataset ./output --batch-size 50 --max-workers 8

# Quiet mode for automated processing
python process_batch_transcribe.py basic ./large_dataset ./output --quiet
```

## Output Structure

### Directory Layout
```
output_dir/
├── file1_transcription.json          # Basic transcription
├── file2_diarized.json              # Diarized transcription (JSON)
├── file3_diarized.eaf               # Diarized transcription (EAF)
├── ...
└── batch_summary.json               # Processing summary
```

### Basic Transcription Output
```json
{
  "text": "Transcribed text here",
  "language": "Greek",
  "duration": 10.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 10.5,
      "text": "Transcribed text here"
    }
  ]
}
```

### Diarized Transcription Output
```json
{
  "diarization": {
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "speaker": "SPEAKER_00"
      }
    ]
  },
  "transcription": {
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "text": "Speaker 1 text",
        "speaker": "SPEAKER_00"
      }
    ]
  }
}
```

### Batch Summary
```json
{
  "total_files": 100,
  "successful": 98,
  "failed": 2,
  "total_time": 1200.5,
  "total_duration": 3600.0,
  "processing_speed": 3.0,
  "results": [...]
}
```

## Performance Optimization

### 1. Model Size Selection

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | 39M | Fastest | Lower | Quick processing, testing |
| base | 74M | Fast | Good | Balanced speed/accuracy |
| small | 244M | Medium | Better | Production use |
| medium | 769M | Slow | High | High accuracy needed |
| large | 1550M | Slowest | Best | Maximum accuracy |

### 2. Parallel Processing Tuning

```bash
# For CPU-bound tasks (basic transcription)
--max-workers 8  # Use all CPU cores

# For GPU-bound tasks (diarized transcription)
--max-workers 2  # Limit to avoid GPU memory issues

# For memory-constrained systems
--max-workers 1 --batch-size 5
```

### 3. Batch Size Guidelines

- **Small files (< 1MB)**: `--batch-size 20-50`
- **Medium files (1-10MB)**: `--batch-size 10-20`
- **Large files (> 10MB)**: `--batch-size 5-10`
- **Memory limited**: `--batch-size 1-5`

## Troubleshooting

### Common Issues

#### 1. Out of Memory
```bash
# Reduce batch size and workers
python process_batch_transcribe.py basic ./audio ./output --batch-size 5 --max-workers 2
```

#### 2. Slow Processing
```bash
# Increase parallel workers
python process_batch_transcribe.py basic ./audio ./output --max-workers 8

# Use smaller model
python process_batch_transcribe.py basic ./audio ./output --model-size tiny
```

#### 3. Audio Format Issues
The script automatically handles format conversion, but if issues persist:
```bash
# Convert files to WAV first
for file in *.mp3; do ffmpeg -i "$file" "${file%.mp3}.wav"; done
```

#### 4. Permission Errors
```bash
# Ensure output directory is writable
mkdir -p ./output
chmod 755 ./output
```

### Performance Monitoring

```bash
# Monitor system resources during processing
htop  # or top

# Check disk space
df -h

# Monitor GPU usage (if using GPU)
nvidia-smi
```

## Integration with Other Scripts

### With Dataset Preparation
```bash
# 1. Batch transcribe audio files
python process_batch_transcribe.py diarized ./raw_audio ./transcriptions --output-format eaf

# 2. Prepare dataset from transcriptions
python process_prepair_datasets.py --input ./transcriptions --dataset-url your-username/greek-dataset
```

### With Model Training
```bash
# 1. Batch transcribe training data
python process_batch_transcribe.py basic ./training_audio ./training_transcriptions

# 2. Train model
python process_train.py --language Greek --language_code el --dataset your-username/greek-dataset
```

## Tips for Best Performance

1. **Use SSD storage** for faster I/O
2. **Ensure sufficient RAM** (8GB+ recommended)
3. **Use appropriate model size** for your accuracy needs
4. **Monitor system resources** during processing
5. **Process in batches** for large datasets
6. **Use quiet mode** for automated processing
7. **Check output quality** with a small batch first

## License

This script is part of the Greek-Whisper project and follows the same license terms.
