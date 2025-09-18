# process_prepair_datasets.py — Dataset Preparation CLI Guide

Prepare a Hugging Face audio dataset from an annotation file (JSON or EAF). The script will:
- Convert EAF → repository JSON template (internally)
- Segment the source audio into train/test WAV files
- Generate metadata files (CSV) per split and combined
- Build and push an Audio dataset to the Hugging Face Hub

## Prerequisites

- Install dependencies and login to the Hub:
```
pip install -r requirements.txt
huggingface-cli login
```

## Command Overview

```
python process_prepair_datasets.py --input <file.eaf|file.json> --dataset-url <user/repo> [--output-dir ...] [--train-prefix ...] [--test-prefix ...] [--segment-prefix ...] [--verbose]
```

## Required Arguments

- `--input`: Annotation file (.json or .eaf)
- `--dataset-url`: Target Hugging Face repo, e.g. `username/Greek-Datasets`

## Optional Arguments

- `--output-dir`: Directory for segments and metadata (default: `data/dataset_output`)
- `--train-prefix`: Train split subdirectory name (default: `train`)
- `--test-prefix`: Test split subdirectory name (default: `test`)
- `--segment-prefix`: Filename prefix for segments (default: `segment`)
- `--verbose`: Print extra logs

## Examples

### From JSON annotations
```
python process_prepair_datasets.py \
  --input data/test2_transcribed.json \
  --dataset-url aImonster111/Greek-Datasets \
  --output-dir data/test2_dataset \
  --verbose
```

### From EAF annotations (auto-conversion)
```
python process_prepair_datasets.py \
  --input data/test2_transcribed.eaf \
  --dataset-url aImonster111/Greek-Datasets \
  --output-dir data/test2_dataset \
  --verbose
```

## Output Layout

```
<output-dir>/
  train/
    segment_0001_...wav
    ...
    metadata.csv
  test/
    segment_0002_...wav
    ...
    metadata.csv
  metadata.csv        # combined (if enabled)
```

Metadata fields include at least: `file_name`, `text`, and `split` (others may be present depending on your configuration).

## Notes & Tips

- Ensure the annotation JSON’s `audio_url` (or the EAF-converted JSON) points to an existing audio file; relative paths typically reside under the `data/` folder.
- The script calls `training.dataset.json2dataset(...)`, which performs segmentation, metadata creation, and pushes the dataset via `push_to_hub`.
- Make sure you have permissions to push to the target Hugging Face repo.
