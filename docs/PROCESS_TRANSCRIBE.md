# process_transcribe.py — Transcription CLI Guide

This CLI transcribes Greek audio using Whisper with optional diarization. Output is saved based on the file extension you pass to `--output`.

## Prerequisites

- Install project dependencies:
```
pip install -r requirements.txt
```
- Make sure PyTorch, torchaudio, and Whisper are installed per your hardware.

## Command Overview

```
python process_transcribe.py --input_file <audio> [--model-size ...] [--language ...] --output <path> <command>
```

Commands:
- `basic`: Simple Whisper transcription (plain text output)
- `diarized`: Whisper + speaker diarization (save `.json` or `.eaf`)
- `advanced`: Diarized transcription with detailed console logging (save `.json` or `.eaf`)

Supported audio formats: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`

## Options

- `--input_file`: Path to the input audio file
- `--model-size`: `tiny | base | small | medium | large` (default: `base`)
- `--language`: Language for transcription (default: `Greek`)
- `--output`:
  - For `basic`: Use a text file (e.g., `out.txt`)
  - For `diarized`/`advanced`: Must be `.json` or `.eaf`
- `--verbose`: Print stack traces on unexpected errors

## Usage Examples

### Basic (plain text)
```
python process_transcribe.py --input_file data/test1.wav --model-size base --language Greek --output data/out.txt basic 
```

### Diarized (ELAN EAF)
```
python process_transcribe.py --input_file data/test1.wav --model-size large --language Greek --output data/out.eaf diarized 
```

### Diarized (JSON)
```
python process_transcribe.py --input_file data/test1.wav --model-size large --language Greek --output data/out.json diarized 
```

### Advanced (detailed console logging)
```
python process_transcribe.py --input_file data/test1.wav --model-size base --language Greek --output data/detailed.json advanced 
```

## Outputs

- `basic` → plain text file containing the transcription
- `diarized` / `advanced` → structured annotation with tiers and segments
  - `.json`: template-style JSON used in this repo
  - `.eaf`: ELAN Annotation Format, viewable in ELAN

## Troubleshooting

- "unrecognized arguments": Ensure flags appear before or after the command as designed by the CLI (this repository defines global flags; follow the examples exactly).
- "Audio file not found": Check your path and extension.
- GPU/VRAM issues: try `--model-size base` or `small`.
- Diarization dependencies: Ensure `pyannote.audio` is installed.

## Notes

- The CLI writes to the `--output` path and determines format by file extension for diarized/advanced.
- The underlying transcribers return data only; saving is handled by the CLI.
