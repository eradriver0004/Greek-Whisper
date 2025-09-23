#!/usr/bin/env python3
"""
Batch Audio Transcription Script

This script processes multiple audio files in batches for faster transcription.
Supports both basic and diarized transcription with parallel processing.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import json
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.transcribe import WhisperTranscriber, DiarizedTranscriber
from utils.utils import write_json, write_eaf


def transcribe_single_file(args_tuple):
    """Transcribe a single audio file - designed for multiprocessing"""
    (audio_path, output_path, model_size, language, transcription_type, verbose) = args_tuple
    
    try:
        if verbose:
            print(f"Processing: {audio_path}")
        
        if transcription_type == "basic":
            transcriber = WhisperTranscriber(model_size=model_size, language=language)
            result = transcriber.transcribe_file(audio_path)
            
            # Save basic transcription
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        elif transcription_type == "diarized":
            transcriber = DiarizedTranscriber(model_size=model_size, language=language)
            result = transcriber.transcribe_file_with_diarization(audio_path)
            
            # Save diarized transcription based on file extension
            if output_path.endswith('.eaf'):
                write_eaf(result, output_path)
            else:
                write_json(result, output_path)
        
        if verbose:
            print(f"✓ Completed: {output_path}")
        
        return {
            'input': audio_path,
            'output': output_path,
            'status': 'success',
            'duration': result.get('duration', 0) if isinstance(result, dict) else 0
        }
        
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {str(e)}"
        print(f"✗ {error_msg}")
        return {
            'input': audio_path,
            'output': output_path,
            'status': 'error',
            'error': str(e)
        }


def find_audio_files(input_path, extensions=None):
    """Find all audio files in the given path"""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
    
    audio_files = []
    
    if os.path.isfile(input_path):
        # Single file
        if any(input_path.lower().endswith(ext) for ext in extensions):
            audio_files.append(input_path)
    elif os.path.isdir(input_path):
        # Directory - find all audio files
        for ext in extensions:
            pattern = os.path.join(input_path, f"**/*{ext}")
            audio_files.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(input_path, f"**/*{ext.upper()}")
            audio_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(audio_files)


def create_output_path(input_path, output_dir, transcription_type, output_format):
    """Create output path for transcription result"""
    input_name = Path(input_path).stem
    
    if transcription_type == "basic":
        return os.path.join(output_dir, f"{input_name}_transcription.json")
    else:  # diarized
        if output_format == "eaf":
            return os.path.join(output_dir, f"{input_name}_diarized.eaf")
        else:
            return os.path.join(output_dir, f"{input_name}_diarized.json")


def batch_transcribe(input_path, output_dir, model_size="base", language="Greek", 
                    transcription_type="basic", output_format="json", 
                    batch_size=None, max_workers=None, verbose=True):
    """
    Process multiple audio files in batches
    
    Args:
        input_path: Path to audio file or directory containing audio files
        output_dir: Directory to save transcription results
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language for transcription
        transcription_type: "basic" or "diarized"
        output_format: "json" or "eaf" (for diarized only)
        batch_size: Number of files to process in each batch (None = process all)
        max_workers: Maximum number of parallel workers (None = auto)
        verbose: Print progress information
    """
    
    # Find all audio files
    audio_files = find_audio_files(input_path)
    
    if not audio_files:
        print("No audio files found!")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up multiprocessing
    if max_workers is None:
        max_workers = min(cpu_count(), len(audio_files))
    
    if batch_size is None:
        batch_size = len(audio_files)
    
    print(f"Processing with {max_workers} workers, batch size: {batch_size}")
    
    # Prepare arguments for each file
    file_args = []
    for audio_file in audio_files:
        output_file = create_output_path(audio_file, output_dir, transcription_type, output_format)
        file_args.append((
            audio_file, output_file, model_size, language, 
            transcription_type, verbose
        ))
    
    # Process in batches
    results = []
    total_processed = 0
    total_duration = 0
    
    start_time = time.time()
    
    for i in range(0, len(file_args), batch_size):
        batch = file_args[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(file_args) + batch_size - 1) // batch_size
        
        print(f"\n--- Processing Batch {batch_num}/{total_batches} ({len(batch)} files) ---")
        
        batch_start = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks in this batch
            future_to_args = {executor.submit(transcribe_single_file, args): args for args in batch}
            
            # Process completed tasks
            for future in as_completed(future_to_args):
                result = future.result()
                results.append(result)
                total_processed += 1
                
                if result['status'] == 'success':
                    total_duration += result.get('duration', 0)
                
                if verbose:
                    status_icon = "✓" if result['status'] == 'success' else "✗"
                    print(f"{status_icon} [{total_processed}/{len(file_args)}] {os.path.basename(result['input'])}")
        
        batch_time = time.time() - batch_start
        print(f"Batch {batch_num} completed in {batch_time:.1f}s")
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"BATCH TRANSCRIPTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Average time per file: {total_time/len(results):.1f}s")
    print(f"Total audio duration: {total_duration:.1f}s")
    print(f"Processing speed: {total_duration/total_time:.1f}x real-time")
    
    # Save results summary
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(results),
            'successful': successful,
            'failed': failed,
            'total_time': total_time,
            'total_duration': total_duration,
            'processing_speed': total_duration/total_time if total_time > 0 else 0,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    
    if failed > 0:
        print(f"\nFailed files:")
        for result in results:
            if result['status'] == 'error':
                print(f"  - {result['input']}: {result['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch Audio Transcription Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription of all audio files in a directory
  python process_batch_transcribe.py basic /path/to/audio/files /path/to/output

  # Diarized transcription with EAF output
  python process_batch_transcribe.py diarized /path/to/audio/files /path/to/output --output-format eaf

  # Process with specific model and language
  python process_batch_transcribe.py basic /path/to/audio/files /path/to/output --model-size large --language Greek

  # Process in smaller batches with limited workers
  python process_batch_transcribe.py diarized /path/to/audio/files /path/to/output --batch-size 5 --max-workers 2
        """
    )
    
    subparsers = parser.add_subparsers(dest='transcription_type', help='Transcription type')
    
    # Basic transcription
    basic_parser = subparsers.add_parser('basic', help='Basic transcription')
    basic_parser.add_argument('input_path', help='Path to audio file or directory')
    basic_parser.add_argument('output_dir', help='Output directory for transcriptions')
    
    # Diarized transcription
    diarized_parser = subparsers.add_parser('diarized', help='Diarized transcription')
    diarized_parser.add_argument('input_path', help='Path to audio file or directory')
    diarized_parser.add_argument('output_dir', help='Output directory for transcriptions')
    diarized_parser.add_argument('--output-format', choices=['json', 'eaf'], default='json',
                                help='Output format for diarized transcription')
    
    # Common arguments
    for subparser in [basic_parser, diarized_parser]:
        subparser.add_argument('--model-size', 
                             choices=['tiny', 'base', 'small', 'medium', 'large'], 
                             default='base',
                             help='Whisper model size')
        subparser.add_argument('--language', default='Greek',
                             help='Language for transcription')
        subparser.add_argument('--batch-size', type=int, default=None,
                             help='Number of files to process in each batch (default: process all)')
        subparser.add_argument('--max-workers', type=int, default=None,
                             help='Maximum number of parallel workers (default: auto)')
        subparser.add_argument('--verbose', action='store_true', default=True,
                             help='Print progress information')
        subparser.add_argument('--quiet', action='store_true',
                             help='Suppress progress output')
    
    args = parser.parse_args()
    
    if not args.transcription_type:
        parser.print_help()
        return
    
    # Handle quiet flag
    verbose = args.verbose and not getattr(args, 'quiet', False)
    
    # Run batch transcription
    batch_transcribe(
        input_path=args.input_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        language=args.language,
        transcription_type=args.transcription_type,
        output_format=getattr(args, 'output_format', 'json'),
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        verbose=verbose
    )


if __name__ == "__main__":
    main()
