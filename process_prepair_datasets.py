#!/usr/bin/env python3
"""
Prepare datasets from annotation files (EAF or JSON) and push to Hugging Face.

Usage examples:
  # From JSON annotation
  python process_prepair_datasets.py --input data/test2_transcribed.json --dataset-url aImonster111/Greek-Datasets --output-dir data/test2_dataset

  # From EAF annotation (will be converted to JSON template internally)
  python process_prepair_datasets.py --input data/test2_transcribed.eaf --dataset-url aImonster111/Greek-Datasets --output-dir data/test2_dataset
"""

import argparse
import os
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio datasets from annotations and push to Hugging Face",
    )

    parser.add_argument("--input", required=True, help="Path to input annotation file (.json or .eaf)")
    parser.add_argument("--dataset-url", required=True, help="Hugging Face dataset repo, e.g. username/repo")
    parser.add_argument("--output-dir", default="data/dataset_output", help="Directory to write segments and metadata")
    parser.add_argument("--train-prefix", default="train", help="Subdirectory name for train split")
    parser.add_argument("--test-prefix", default="test", help="Subdirectory name for test split")
    parser.add_argument("--segment-prefix", default="segment", help="Filename prefix for segment WAVs")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Normalize paths
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Add project root to path so we can import modules
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import helpers
    from utils.utils import read_eaf, write_json
    from training.dataset import json2dataset

    # Prepare a JSON path for json2dataset
    ext = os.path.splitext(input_path)[1].lower()
    json_path = None

    if ext == ".json":
        json_path = input_path
    elif ext == ".eaf":
        if args.verbose:
            print(f"Reading EAF and converting to JSON template: {input_path}")
        data = read_eaf(input_path)
        # Save to a temporary JSON file
        tmp_dir = tempfile.mkdtemp(prefix="prep_ds_")
        json_path = os.path.join(tmp_dir, "converted_from_eaf.json")
        write_json(data, json_path)
        if args.verbose:
            print(f"Converted JSON saved to: {json_path}")
    else:
        raise ValueError("Unsupported input file type. Expected .json or .eaf")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if args.verbose:
        print(f"Creating dataset from: {json_path}")
        print(f"Output directory: {output_dir}")
        print(f"Train/Test prefixes: {args.train_prefix}/{args.test_prefix}")
        print(f"Segment prefix: {args.segment_prefix}")
        print(f"Pushing to: {args.dataset_url}")

    # Build dataset and push
    dataset = json2dataset(
        json_path,
        output_dir=output_dir,
        segment_prefix=args.segment_prefix,
        train_prefix=args.train_prefix,
        test_prefix=args.test_prefix,
        dataset_url=args.dataset_url,
    )

    # Note: json2dataset handles creating metadata files and pushing to hub
    if args.verbose:
        try:
            from datasets import DatasetDict
            if isinstance(dataset, DatasetDict):
                print(f"✓ Dataset pushed. Splits: {list(dataset.keys())}")
            else:
                print("✓ Dataset pushed.")
        except Exception:
            print("✓ Dataset pushed.")

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())


