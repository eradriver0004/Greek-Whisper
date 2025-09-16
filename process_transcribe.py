#!/usr/bin/env python3
"""
Process Transcribe - Command-line interface for Greek audio transcription

This script provides a comprehensive interface for transcribing Greek audio files
using various methods including basic Whisper, diarization, and speaker separation.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the current directory to the path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.transcribe import (
    WhisperTranscriber,
    DiarizedTranscriber
)
from utils.utils import write_json, write_eaf

def validate_audio_file(file_path):
    """Validate that the audio file exists and is readable."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check if it's a valid audio file by extension
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"Warning: {file_path} may not be a supported audio format")
    
    return True

def basic_transcribe(args):
    """Perform basic transcription using Whisper."""
    print(f"Starting basic transcription of {args.input_file}...")
    
    transcriber = WhisperTranscriber(
        model_size=args.model_size,
        language=args.language
    )
    
    # Load audio file
    import librosa
    audio_data, sr = librosa.load(args.input_file, sr=16000)
    
    # Transcribe
    start_time = time.time()
    result = transcriber.transcribe(audio_data)
    end_time = time.time()
    
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")
    print(f"Result: {result}")
    
    # Save result if output file specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Transcription saved to: {args.output}")

def diarized_transcribe(args):
    """Perform diarized transcription."""
    print(f"Starting diarized transcription of {args.input_file}...")
    
    transcriber = DiarizedTranscriber(
        model_size=args.model_size,
        language=args.language
    )
    
    start_time = time.time()
    # Run diarized transcription (let underlying implementation optionally write defaults)
    result = transcriber.transcribe_file_with_diarization(args.input_file)
    end_time = time.time()
    
    print(f"Diarized transcription completed in {end_time - start_time:.2f} seconds")
    print(f"Found {len(result.get('tiers', []))} speakers")
    
    # Save result via --output path if provided
    if args.output:
        out_lower = args.output.lower()
        if out_lower.endswith('.json'):
            write_json(result, args.output)
            print(f"Saved JSON to: {args.output}")
        elif out_lower.endswith('.eaf'):
            write_eaf(result, args.output)
            print(f"Saved EAF to: {args.output}")
        else:
            raise ValueError("--output must end with .json or .eaf for diarized/advanced modes")

    # Print sample transcriptions
    for tier_id in result.get('tiers', []):
        if tier_id in result and 'annotations' in result[tier_id]:
            annotations = result[tier_id]['annotations']
            if annotations:
                print(f"{tier_id}: {annotations[0]['value']}")

def advanced_transcribe(args):
    """Perform advanced transcription with detailed speaker analysis."""
    print(f"Starting advanced transcription of {args.input_file}...")
    
    transcriber = DiarizedTranscriber(
        model_size=args.model_size,
        language=args.language
    )
    
    start_time = time.time()
    result = transcriber.transcribe_file_with_diarization(args.input_file)
    end_time = time.time()
    
    print(f"Advanced transcription completed in {end_time - start_time:.2f} seconds")
    print(f"Found {len(result.get('tiers', []))} speakers")
    
    # Save result via --output path if provided
    if args.output:
        out_lower = args.output.lower()
        if out_lower.endswith('.json'):
            write_json(result, args.output)
            print(f"Saved JSON to: {args.output}")
        elif out_lower.endswith('.eaf'):
            write_eaf(result, args.output)
            print(f"Saved EAF to: {args.output}")
        else:
            raise ValueError("--output must end with .json or .eaf for diarized/advanced modes")

    # Print detailed speaker information
    for tier_id in result.get('tiers', []):
        if tier_id in result and 'annotations' in result[tier_id]:
            annotations = result[tier_id]['annotations']
            print(f"\n{tier_id}: {len(annotations)} segments")
            
            # Show first few transcriptions
            for i, annotation in enumerate(annotations[:3]):
                start_time_ms = int(annotation['start_time'])
                end_time_ms = int(annotation['end_time'])
                duration = (end_time_ms - start_time_ms) / 1000.0
                print(f"  {i+1}. [{start_time_ms/1000:.1f}s - {end_time_ms/1000:.1f}s] ({duration:.1f}s): {annotation['value']}")
            
            if len(annotations) > 3:
                print(f"  ... and {len(annotations) - 3} more segments")

def main():
    """Main function to handle command-line arguments and execute transcription."""
    parser = argparse.ArgumentParser(
        description="Greek Audio Transcription Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python process_transcribe.py basic data/test.wav --model-size base --language Greek

  # Diarized transcription
  python process_transcribe.py diarized data/test.wav --output data/test_output.json

  # Advanced transcription with detailed analysis
  python process_transcribe.py advanced data/test.wav --output data/detailed_output.eaf
        """
    )
    
    # Common arguments
    parser.add_argument(
        "--input_file",
        help="Path to the input audio file"
    )
    
    parser.add_argument(
        "--model-size",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)"
    )
    
    parser.add_argument(
        "--language",
        default="Greek",
        help="Language for transcription (default: Greek)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path. For diarized/advanced use .json or .eaf extension"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Transcription method")
    
    # Basic transcription
    basic_parser = subparsers.add_parser(
        "basic",
        help="Basic Whisper transcription"
    )
    
    # Diarized transcription
    diarized_parser = subparsers.add_parser(
        "diarized",
        help="Transcription with speaker diarization"
    )
    
    # Advanced transcription
    advanced_parser = subparsers.add_parser(
        "advanced",
        help="Advanced transcription with detailed speaker analysis"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Validate input file
        validate_audio_file(args.input_file)
        
        # Execute the appropriate transcription method
        if args.command == "basic":
            basic_transcribe(args)
        elif args.command == "diarized":
            diarized_transcribe(args)
        elif args.command == "advanced":
            advanced_transcribe(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
