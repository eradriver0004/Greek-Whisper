import json
import os
import sys
import soundfile as sf
from datasets import Dataset, Audio, load_dataset
from pathlib import Path

import librosa
import csv

# Add parent directory to path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from utils.utils import file_url_to_path
except ImportError:
    # Fallback: try importing directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("utils", os.path.join(parent_dir, "utils", "utils.py"))
    utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_module)
    file_url_to_path = utils_module.file_url_to_path

def json2dataset(json_path, output_dir="segmented_audio", segment_prefix="segment", train_prefix="train", test_prefix="test", dataset_url="aImonster111/Greek-Datasets"):
    """
    Convert JSON annotation data to a dataset with segmented audio files.
    
    Args:
        json_path (str): Path to the JSON file containing annotations
        output_dir (str): Directory to save segmented audio files
        segment_prefix (str): Prefix for segmented audio filenames
        
    Returns:
        Dataset: Hugging Face dataset with audio file paths and metadata
    """
    with open(json_path, 'r') as f:
        data = json.load(f) 

    # Load the full audio file
    audio_path = file_url_to_path(data['audio_url'])
    print(f"Audio path: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, train_prefix), exist_ok=True)
    os.makedirs(os.path.join(output_dir, test_prefix), exist_ok=True)
    
    # Prepare dataset entries and metadata
    dataset_audio_entries = []
    metadata_entries = []
    segment_counter = 1
    
    # Process each tier (speaker)
    for tier_id in data.get('tiers', []):
        if tier_id in data and 'annotations' in data[tier_id]:
            tier_data = data[tier_id]
            annotations = tier_data['annotations']
            
            # Process each annotation (segment)
            for annotation in annotations:
                try:
                    # Extract timing information (convert from milliseconds to samples)
                    start_time_ms = int(annotation['start_time'])
                    end_time_ms = int(annotation['end_time'])
                    
                    # Convert to sample indices
                    start_sample = int((start_time_ms / 1000.0) * sr)
                    end_sample = int((end_time_ms / 1000.0) * sr)
                    
                    # Ensure we don't go out of bounds
                    start_sample = max(0, start_sample)
                    end_sample = min(len(audio), end_sample)
                    
                    # Extract audio segment
                    audio_segment = audio[start_sample:end_sample]
                    
                    # Skip empty segments
                    if len(audio_segment) == 0:
                        print(f"Warning: Empty segment for {tier_id} annotation {annotation['id']}")
                        continue
                    
                    # Create filename for this segment
                    segment_filename = f"{segment_prefix}_{segment_counter:04d}_{tier_id}_{annotation['id']}.wav"

                    if segment_counter % 3 != 2:
                        segment_path = os.path.join(output_dir, train_prefix, segment_filename)
                    else:
                        segment_path = os.path.join(output_dir, test_prefix, segment_filename)
                    
                    # Save the audio segment
                    sf.write(segment_path, audio_segment, sr, format='WAV', subtype='PCM_16')
                    
                    # Create dataset entry
                    dataset_audio_entries.append({"audio": segment_path})
                    
                    # Create metadata entry
                    metadata_entry = {
                        "file_name": segment_filename,
                        "text": annotation['value'],  # Transcription text
                        "split": train_prefix if segment_counter % 3 != 2 else test_prefix
                    }
                    metadata_entries.append(metadata_entry)
                    
                    segment_counter += 1
                    
                    print(f"✓ Created segment: {segment_filename} ({len(audio_segment)/sr:.2f}s)")
                    
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid annotation {annotation.get('id', 'unknown')}: {e}")
                    continue
    
    # Create Hugging Face dataset
    print("dataset_audio_entries: ", dataset_audio_entries)
    if dataset_audio_entries:
        # Create metadata files
        create_metadata_files(metadata_entries, output_dir, train_prefix, test_prefix)

        # Create a list of dictionaries, each with an "audio" key
        dataset = load_dataset("audiofolder", data_dir=output_dir)
        
        # push to hub
        dataset.push_to_hub(dataset_url)
        
        print(f"\n✓ Created dataset with {len(dataset_audio_entries)} audio segments")
        print(f"✓ Segments saved to: {output_dir}")
        
        return dataset
    else:
        print("Warning: No valid audio segments found")
        return Dataset.from_list([])

def create_metadata_files(metadata_entries, output_dir, train_prefix, test_prefix):
    """
    Create metadata files (CSV, JSONL, Parquet) for the audio dataset.
    
    Args:
        metadata_entries (list): List of metadata dictionaries
        output_dir (str): Output directory for metadata files
        train_prefix (str): Train split prefix
        test_prefix (str): Test split prefix
    """
    if not metadata_entries:
        print("Warning: No metadata entries to create files")
        return
    
    # Create metadata files for each split
    for split_prefix in [train_prefix, test_prefix]:
        split_entries = [entry for entry in metadata_entries if entry['split'] == split_prefix]
        
        if not split_entries:
            continue
            
        split_dir = os.path.join(output_dir, split_prefix)
        
        # Create CSV file
        csv_path = os.path.join(split_dir, "metadata.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = split_entries[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_entries)
        
        print(f"✓ Created metadata files for {split_prefix}: {len(split_entries)} entries")

if __name__ == "__main__":
    # json_path = "data/test2_transcribed.json"
    # output_dir = "data/test2_dataset"
    # json2dataset(json_path, output_dir)
    dataset_url = "aImonster111/Greek-Datasets"
    dataset = load_dataset(dataset_url)
    print(dataset)