import whisper
import numpy as np
import librosa
import os
import torch
import torchaudio
from abc import ABC, abstractmethod

# Import separation functionality
try:
    from utils.separation import SpeakerSeparator, AudioProcessor
except ImportError:
    # Handle case when running script directly
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from separation import SpeakerSeparator, AudioProcessor

class BaseTranscriber(ABC):
    """Base class for all transcription functionality."""
    
    def __init__(self, model_size="large", language=None):
        self.model_size = model_size
        self.language = language
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            self.model = whisper.load_model(self.model_size)
    
    @abstractmethod
    def transcribe(self, audio_data):
        """Transcribe audio data. Must be implemented by subclasses."""
        pass


class WhisperTranscriber(BaseTranscriber):
    """Basic Whisper transcription functionality."""
    
    def transcribe(self, audio_data):
        """
        Transcribe audio sequence data to text using Whisper.
        
        Args:
            audio_data (numpy.ndarray): Audio data as numpy array
            
        Returns:
            str: Transcribed text
        """
        
        # Transcribe the audio data
        result = self.model.transcribe(
            audio_data,
            language=self.language,
            verbose=False
        )
        
        return result["text"].strip()

class DiarizedTranscriber(BaseTranscriber):
    """Transcriber that handles diarization and transcription."""
    
    def __init__(self, model_size="large", language=None):
        super().__init__(model_size, language)
        self.diarization_data = None
    
    def diarize_audio(self, audio_path):
        """Perform diarization on audio file."""
        try:
            from utils.diarization import diarize
        except ImportError:
            # Handle case when running script directly
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            from diarization import diarize
        
        print("Performing diarization...")
        self.diarization_data = diarize(audio_path)
        return self.diarization_data
    
    def transcribe(self, audio_data):
        """Transcribe audio data (basic implementation)."""
        return super().transcribe(audio_data)
    
    def transcribe_file_with_diarization(self, audio_path, output_prefix=None):
        """
        Diarize audio file and transcribe each speaker segment individually.
        
        Args:
            audio_path (str): Path to the audio file
            output_prefix (str, optional): Prefix for output files. If None, uses audio filename
            
        Returns:
            dict: Diarization data with transcribed text in each annotation
        """
        # Perform diarization
        diarization_data = self.diarize_audio(audio_path)
        
        # Load audio file
        print("Loading audio file...")
        audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
        
        # Load Whisper model
        self.load_model()
        
        # Transcribe each annotation segment and remove failed ones
        print("Transcribing segments...")
        annotations_to_remove = []
        
        for tier_id in diarization_data.get("tiers", []):
            if tier_id in diarization_data and "annotations" in diarization_data[tier_id]:
                annotations = diarization_data[tier_id]["annotations"]
                
                for i, annotation in enumerate(annotations):
                    start_time = int(annotation["start_time"]) / 1000.0  # Convert to seconds
                    end_time = int(annotation["end_time"]) / 1000.0      # Convert to seconds
                    
                    # Extract audio segment
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    audio_segment = audio[start_sample:end_sample]
                    
                    # Transcribe the segment
                    if len(audio_segment) > 0:
                        result = self.model.transcribe(
                            audio_segment,
                            language=self.language,
                            verbose=False
                        )
                        transcribed_text = result["text"].strip()
                        annotation["value"] = transcribed_text
                        
                        # Check if transcription failed (empty or very short text)
                        if len(transcribed_text) < 2:  # Less than 2 characters
                            print(f"Removing failed transcription for {tier_id} segment: '{transcribed_text}'")
                            annotations_to_remove.append((tier_id, i))
                        else:
                            print(f"Transcribed {tier_id} segment: {transcribed_text[:50]}...")
                    else:
                        print(f"Removing empty audio segment for {tier_id}")
                        annotations_to_remove.append((tier_id, i))
        
        # Remove failed annotations (in reverse order to maintain indices)
        for tier_id, index in reversed(annotations_to_remove):
            if tier_id in diarization_data and "annotations" in diarization_data[tier_id]:
                del diarization_data[tier_id]["annotations"][index]
        
        # Clean up empty tiers and update last_used_annotation_id
        try:
            from utils.utils import remove_annotation_by_id
        except ImportError:
            from utils import remove_annotation_by_id
        
        # Remove empty tiers
        tiers_to_remove = []
        for tier_id in diarization_data.get("tiers", []):
            if tier_id in diarization_data and "annotations" in diarization_data[tier_id]:
                if not diarization_data[tier_id]["annotations"]:
                    tiers_to_remove.append(tier_id)
        
        for tier_id in tiers_to_remove:
            if tier_id in diarization_data:
                del diarization_data[tier_id]
            if tier_id in diarization_data.get("tiers", []):
                diarization_data["tiers"].remove(tier_id)
        
        # Update last_used_annotation_id
        max_id = 0
        for tier in diarization_data.get("tiers", []):
            if tier in diarization_data and "annotations" in diarization_data[tier]:
                for ann in diarization_data[tier]["annotations"]:
                    ann_id = ann.get("id", "")
                    if ann_id.startswith("a"):
                        try:
                            ann_num = int(ann_id[1:])
                            max_id = max(max_id, ann_num)
                        except ValueError:
                            pass
        diarization_data["last_used_annotation_id"] = max_id
        
        # Save results
        if output_prefix is None:
            output_prefix = os.path.splitext(os.path.basename(audio_path))[0]
        
        json_path = f"data/{output_prefix}_transcribed.json"
        eaf_path = f"data/{output_prefix}_transcribed.eaf"
        
        print(f"Saving results to {json_path} and {eaf_path}")
        
        try:
            from utils.utils import write_json, write_eaf
        except ImportError:
            from utils import write_json, write_eaf
        
        write_json(diarization_data, json_path)
        write_eaf(diarization_data, eaf_path)
        
        return diarization_data


class TranscriptionPipeline:
    """Complete transcription pipeline with speaker separation."""
    
    def __init__(self, model_size="large", language=None):
        self.transcriber = DiarizedTranscriber(model_size, language)
        self.separator = SpeakerSeparator()
    
    def transcribe_with_asteroid_separation(self, audio_path, num_speakers=2, output_prefix=None):
        """
        Complete pipeline: Separate speakers with Asteroid, diarize each, transcribe, and merge.
        
        Args:
            audio_path (str): Path to the audio file
            num_speakers (int): Number of speakers to separate (2 or 3)
            output_prefix (str): Prefix for output files
            
        Returns:
            dict: Merged annotation data with all speakers transcribed
        """
        print(f"Starting complete transcription pipeline for {num_speakers} speakers...")
        
        # Step 1: Separate speakers using Asteroid
        separated_audios, sr = self.separator.separate_speakers_asteroid(audio_path, num_speakers)
        
        # Step 2: Diarize and transcribe each separated audio
        speaker_results = self._diarize_and_transcribe_separated_audio(separated_audios, sr)
        
        # Step 3: Merge all speaker annotations
        merged_data = self._merge_speaker_annotations(speaker_results, audio_path)
        
        # Save results
        if output_prefix is None:
            output_prefix = os.path.splitext(os.path.basename(audio_path))[0]
        
        json_path = f"data/{output_prefix}_asteroid_transcribed.json"
        eaf_path = f"data/{output_prefix}_asteroid_transcribed.eaf"
        
        print(f"Saving results to {json_path} and {eaf_path}")
        
        try:
            from utils.utils import write_json, write_eaf
        except ImportError:
            from utils import write_json, write_eaf
        
        write_json(merged_data, json_path)
        write_eaf(merged_data, eaf_path)
        
        return merged_data
    
    def _diarize_and_transcribe_separated_audio(self, separated_audios, sr):
        """
        Diarize and transcribe each separated audio.
        
        Args:
            separated_audios (list): List of separated audio arrays
            sr (int): Sample rate
            
        Returns:
            list: List of diarization data for each speaker
        """
        try:
            from utils.diarization import diarize
        except ImportError:
            from diarization import diarize
        
        speaker_results = []
        
        for i, audio_data in enumerate(separated_audios):
            print(f"Processing speaker {i+1}...")
            
            # Save temporary audio file for diarization
            temp_file = f"temp_speaker_{i+1}.wav"
            librosa.output.write_wav(temp_file, audio_data, sr)
            
            try:
                # Diarize the separated audio
                diarization_data = diarize(temp_file)
                
                # Load Whisper model
                self.transcriber.load_model()
                
                # Transcribe each annotation segment
                for tier_id in diarization_data.get("tiers", []):
                    if tier_id in diarization_data and "annotations" in diarization_data[tier_id]:
                        annotations = diarization_data[tier_id]["annotations"]
                        
                        for annotation in annotations:
                            start_time = int(annotation["start_time"]) / 1000.0
                            end_time = int(annotation["end_time"]) / 1000.0
                            
                            # Extract audio segment
                            start_sample = int(start_time * sr)
                            end_sample = int(end_time * sr)
                            audio_segment = audio_data[start_sample:end_sample]
                            
                            # Transcribe the segment
                            if len(audio_segment) > 0:
                                result = self.transcriber.model.transcribe(
                                    audio_segment,
                                    language=self.transcriber.language,
                                    verbose=False
                                )
                                transcribed_text = result["text"].strip()
                                annotation["value"] = transcribed_text
                                
                                # Remove failed transcriptions
                                if len(transcribed_text) < 2:
                                    annotation["value"] = ""
                            else:
                                annotation["value"] = ""
                
                # Clean up failed annotations
                annotations_to_remove = []
                for tier_id in diarization_data.get("tiers", []):
                    if tier_id in diarization_data and "annotations" in diarization_data[tier_id]:
                        annotations = diarization_data[tier_id]["annotations"]
                        for j, annotation in enumerate(annotations):
                            if len(annotation.get("value", "")) < 2:
                                annotations_to_remove.append((tier_id, j))
                
                # Remove failed annotations
                for tier_id, index in reversed(annotations_to_remove):
                    if tier_id in diarization_data and "annotations" in diarization_data[tier_id]:
                        del diarization_data[tier_id]["annotations"][index]
                
                # Clean up empty tiers
                tiers_to_remove = []
                for tier_id in diarization_data.get("tiers", []):
                    if tier_id in diarization_data and "annotations" in diarization_data[tier_id]:
                        if not diarization_data[tier_id]["annotations"]:
                            tiers_to_remove.append(tier_id)
                
                for tier_id in tiers_to_remove:
                    if tier_id in diarization_data:
                        del diarization_data[tier_id]
                    if tier_id in diarization_data.get("tiers", []):
                        diarization_data["tiers"].remove(tier_id)
                
                speaker_results.append(diarization_data)
                print(f"Speaker {i+1} processed: {len(diarization_data.get('tiers', []))} tiers")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return speaker_results
    
    def _merge_speaker_annotations(self, speaker_results, original_audio_path):
        """
        Merge multiple speaker annotation results into one template-style data structure.
        
        Args:
            speaker_results (list): List of diarization data for each speaker
            original_audio_path (str): Path to the original audio file
            
        Returns:
            dict: Merged annotation data in template style
        """
        print("Merging speaker annotations...")
        
        merged_data = {
            "audio_url": f"file:///{os.path.abspath(original_audio_path).replace(os.sep, '/')}",
            "last_used_annotation_id": 0,
            "tiers": []
        }
        
        annotation_id = 1
        
        # Process each speaker's results
        for speaker_idx, speaker_data in enumerate(speaker_results):
            for tier_id in speaker_data.get("tiers", []):
                if tier_id in speaker_data and "annotations" in speaker_data[tier_id]:
                    annotations = speaker_data[tier_id]["annotations"]
                    
                    # Create new tier ID for this speaker
                    new_tier_id = f"SPEAKER_{speaker_idx + 1}"
                    
                    # Initialize tier if it doesn't exist
                    if new_tier_id not in merged_data:
                        merged_data[new_tier_id] = {
                            "id": new_tier_id,
                            "annotations": []
                        }
                        merged_data["tiers"].append(new_tier_id)
                    
                    # Add annotations with updated IDs
                    for annotation in annotations:
                        if len(annotation.get("value", "")) >= 2:  # Only add successful transcriptions
                            new_annotation = {
                                "id": f"a{annotation_id}",
                                "start_time": annotation["start_time"],
                                "end_time": annotation["end_time"],
                                "value": annotation["value"]
                            }
                            merged_data[new_tier_id]["annotations"].append(new_annotation)
                            annotation_id += 1
        
        # Update last_used_annotation_id
        merged_data["last_used_annotation_id"] = annotation_id - 1
        
        # Sort annotations by start time within each tier
        for tier_id in merged_data.get("tiers", []):
            if tier_id in merged_data and "annotations" in merged_data[tier_id]:
                merged_data[tier_id]["annotations"].sort(key=lambda x: int(x["start_time"]))
        
        print(f"Merged {len(merged_data.get('tiers', []))} speakers with {merged_data['last_used_annotation_id']} total annotations")
        return merged_data

# Legacy function wrappers for backward compatibility
def transcribe(audio_data, model_size="large", language=None):
    """
    Transcribe audio sequence data to text using Whisper.
    
    Args:
        audio_data (numpy.ndarray): Audio data as numpy array
        model_size (str): Whisper model size ("tiny", "base", "small", "medium", "large")
        language (str, optional): Language code (e.g., "en", "es", "fr"). If None, auto-detects
        
    Returns:
        str: Transcribed text
    """
    transcriber = WhisperTranscriber(model_size, language)
    return transcriber.transcribe(audio_data)

def transcribe_file_with_diarization(audio_path, output_prefix=None, model_size="large", language=None):
    """
    Diarize audio file and transcribe each speaker segment individually.
    
    Args:
        audio_path (str): Path to the audio file
        output_prefix (str, optional): Prefix for output files. If None, uses audio filename
        model_size (str): Whisper model size
        language (str, optional): Language code
        
    Returns:
        dict: Diarization data with transcribed text in each annotation
    """
    transcriber = DiarizedTranscriber(model_size, language)
    return transcriber.transcribe_file_with_diarization(audio_path, output_prefix)

def chunk_audio(audio_tensor, chunk_length_seconds=10, overlap_seconds=1, sample_rate=8000):
    """
    Break audio into overlapping chunks for processing.
    
    Args:
        audio_tensor (torch.Tensor): Audio tensor of shape [1, time]
        chunk_length_seconds (float): Length of each chunk in seconds
        overlap_seconds (float): Overlap between chunks in seconds
        sample_rate (int): Sample rate of the audio
        
    Returns:
        list: List of audio chunks as tensors
        list: List of start times for each chunk in samples
    """
    try:
        from utils.separation import chunk_audio as sep_chunk_audio
    except ImportError:
        from separation import chunk_audio as sep_chunk_audio
    return sep_chunk_audio(audio_tensor, chunk_length_seconds, overlap_seconds, sample_rate)

def stitch_separated_chunks(separated_chunks_list, start_times, overlap_samples, original_length):
    """
    Stitch separated audio chunks back together with overlap handling.
    
    Args:
        separated_chunks_list (list): List of separated chunks, each containing [batch, time, sources]
        start_times (list): List of start times for each chunk
        overlap_samples (int): Number of overlap samples
        original_length (int): Original audio length in samples
        
    Returns:
        torch.Tensor: Stitched separated audio of shape [batch, time, sources]
    """
    try:
        from utils.separation import stitch_separated_chunks as sep_stitch_chunks
    except ImportError:
        from separation import stitch_separated_chunks as sep_stitch_chunks
    return sep_stitch_chunks(separated_chunks_list, start_times, overlap_samples, original_length)

def separate_audio_speechbrain_sepformer(audio_file, out_dir="separated_tracks", chunk_length_seconds=10, overlap_seconds=1):
    """
    Separate audio using SpeechBrain SepformerSeparation with chunking for long audio files.
    
    Args:
        audio_file (str): Path to the audio file
        out_dir (str): Output directory for separated tracks
        chunk_length_seconds (float): Length of each chunk in seconds (5-15 seconds recommended)
        overlap_seconds (float): Overlap between chunks in seconds
        
    Returns:
        list: List of paths to separated audio files
    """
    try:
        from utils.separation import separate_audio_speechbrain_sepformer as sep_speechbrain
    except ImportError:
        from separation import separate_audio_speechbrain_sepformer as sep_speechbrain
    return sep_speechbrain(audio_file, out_dir, chunk_length_seconds, overlap_seconds)

def separate_audio_speechbrain_alternative(audio_file, out_dir="separated_tracks", chunk_length_seconds=10, overlap_seconds=1):
    """
    Alternative SpeechBrain separation using direct model loading without custom savedir, with chunking for long audio files.
    
    Args:
        audio_file (str): Path to the audio file
        out_dir (str): Output directory for separated tracks
        chunk_length_seconds (float): Length of each chunk in seconds (5-15 seconds recommended)
        overlap_seconds (float): Overlap between chunks in seconds
        
    Returns:
        list: List of paths to separated audio files
    """
    try:
        from utils.separation import separate_audio_speechbrain_alternative as sep_speechbrain_alt
    except ImportError:
        from separation import separate_audio_speechbrain_alternative as sep_speechbrain_alt
    return sep_speechbrain_alt(audio_file, out_dir, chunk_length_seconds, overlap_seconds)

def separate_speakers_asteroid(audio_path, num_speakers=2):
    """
    Separate speakers using Asteroid source separation.
    
    Args:
        audio_path (str): Path to the audio file
        num_speakers (int): Number of speakers to separate (2 or 3)
        
    Returns:
        list: List of separated audio arrays for each speaker
    """
    try:
        from utils.separation import separate_speakers_asteroid as sep_asteroid
    except ImportError:
        from separation import separate_speakers_asteroid as sep_asteroid
    return sep_asteroid(audio_path, num_speakers)

def diarize_and_transcribe_separated_audio(separated_audios, sr, model_size="large", language=None):
    """
    Diarize and transcribe each separated audio.
    
    Args:
        separated_audios (list): List of separated audio arrays
        sr (int): Sample rate
        model_size (str): Whisper model size
        language (str): Language code
        
    Returns:
        list: List of diarization data for each speaker
    """
    pipeline = TranscriptionPipeline(model_size, language)
    return pipeline._diarize_and_transcribe_separated_audio(separated_audios, sr)

def merge_speaker_annotations(speaker_results, original_audio_path):
    """
    Merge multiple speaker annotation results into one template-style data structure.
    
    Args:
        speaker_results (list): List of diarization data for each speaker
        original_audio_path (str): Path to the original audio file
        
    Returns:
        dict: Merged annotation data in template style
    """
    pipeline = TranscriptionPipeline()
    return pipeline._merge_speaker_annotations(speaker_results, original_audio_path)

def transcribe_with_asteroid_separation(audio_path, num_speakers=2, output_prefix=None, model_size="large", language=None):
    """
    Complete pipeline: Separate speakers with Asteroid, diarize each, transcribe, and merge.
    
    Args:
        audio_path (str): Path to the audio file
        num_speakers (int): Number of speakers to separate (2 or 3)
        output_prefix (str): Prefix for output files
        model_size (str): Whisper model size
        language (str): Language code
        
    Returns:
        dict: Merged annotation data with all speakers transcribed
    """
    pipeline = TranscriptionPipeline(model_size, language)
    return pipeline.transcribe_with_asteroid_separation(audio_path, num_speakers, output_prefix)

def transcribe_with_speechbrain_sepformer_separation(audio_path, output_prefix=None, model_size="large", language=None):
    """
    Complete pipeline using SpeechBrain SepformerSeparation: Separate speakers, diarize each, transcribe, and merge.
    
    Args:
        audio_path (str): Path to the audio file
        output_prefix (str): Prefix for output files
        model_size (str): Whisper model size
        language (str): Language code
        
    Returns:
        dict: Merged annotation data with all speakers transcribed
    """
    # This function is currently not fully implemented in the class-based approach
    # It would require additional implementation in the TranscriptionPipeline class
    print("SpeechBrain SepformerSeparation pipeline not yet fully implemented in class-based approach")
    print("Falling back to basic diarization and transcription...")
    
    transcriber = DiarizedTranscriber(model_size, language)
    return transcriber.transcribe_file_with_diarization(audio_path, output_prefix)

if __name__ == "__main__":
    import sys
    import os
    # Add the parent directory to the path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    
    # Test the class-based transcription pipeline with separated functionality
    print("Testing class-based transcription pipeline with separated functionality...")
    
    # Example 1: Basic transcription
    print("\n1. Testing basic WhisperTranscriber...")
    # transcriber = WhisperTranscriber(model_size="base", language="Greek")
    # result = transcriber.transcribe(audio_data)  # Would need actual audio data
    
    # Example 2: Diarized transcription
    print("\n2. Testing DiarizedTranscriber...")
    diarized_transcriber = DiarizedTranscriber(model_size="base", language="Greek")
    try:
        result = diarized_transcriber.transcribe_file_with_diarization("data/test2.wav")
        print("Diarization and transcription completed!")
    except FileNotFoundError:
        print("Test file 'data/test2.wav' not found. Please provide a valid audio file.")
    
    # Example 3: Complete pipeline with speaker separation
    # print("\n3. Testing TranscriptionPipeline with Asteroid separation...")
    # pipeline = TranscriptionPipeline(model_size="base", language="Greek")
    # try:
    #     result = pipeline.transcribe_with_asteroid_separation("data/test2.wav", num_speakers=2)
    #     print("Complete pipeline with speaker separation completed!")
    # except FileNotFoundError:
    #     print("Test file 'data/test2.wav' not found. Please provide a valid audio file.")
    # except Exception as e:
    #     print(f"Pipeline test failed: {e}")
    
    # Example 4: Test separation functionality directly
    print("\n4. Testing separation functionality directly...")
    try:
        try:
            from utils.separation import SpeakerSeparator
        except ImportError:
            from separation import SpeakerSeparator
        separator = SpeakerSeparator()
        print("SpeakerSeparator initialized successfully")
        
        # Test Asteroid separation (if audio file exists)
        try:
            separated_audios, sr = separator.separate_speakers_asteroid("data/test2.wav", num_speakers=2)
            print(f"Asteroid separation successful: {len(separated_audios)} speakers separated")
        except FileNotFoundError:
            print("Test file 'data/test2.wav' not found for separation test")
        except Exception as e:
            print(f"Asteroid separation test failed: {e}")
            
    except Exception as e:
        print(f"Separation functionality test failed: {e}")
    
    print("\nClass-based transcription system with separated functionality ready!")
    