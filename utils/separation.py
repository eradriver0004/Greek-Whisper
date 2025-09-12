import os
import torch
import torchaudio
import librosa

# Set SpeechBrain to use copy strategy instead of symlink (fixes Windows issue)
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"

from asteroid.models import ConvTasNet

# Import SpeechBrain with copy strategy
try:
    from speechbrain.inference.separation import SepformerSeparation
except ImportError:
    SepformerSeparation = None


class AudioProcessor:
    """Utility class for audio processing operations."""
    
    @staticmethod
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
        chunk_length_samples = int(chunk_length_seconds * sample_rate)
        overlap_samples = int(overlap_seconds * sample_rate)
        step_size = chunk_length_samples - overlap_samples
        
        chunks = []
        start_times = []
        
        audio_length = audio_tensor.shape[1]
        
        for start_sample in range(0, audio_length, step_size):
            end_sample = min(start_sample + chunk_length_samples, audio_length)
            
            # Skip very short chunks at the end
            if end_sample - start_sample < chunk_length_samples // 2:
                break
                
            chunk = audio_tensor[:, start_sample:end_sample]
            chunks.append(chunk)
            start_times.append(start_sample)
        
        return chunks, start_times
    
    @staticmethod
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
        if not separated_chunks_list:
            return torch.empty(1, original_length, 2)
        
        batch_size = separated_chunks_list[0].shape[0]
        num_sources = separated_chunks_list[0].shape[-1]
        
        # Initialize output tensor
        stitched_audio = torch.zeros(batch_size, original_length, num_sources)
        
        for i, (chunk, start_time) in enumerate(zip(separated_chunks_list, start_times)):
            chunk_length = chunk.shape[1]
            end_time = start_time + chunk_length
            
            if i == 0:
                # First chunk: no overlap handling needed
                stitched_audio[:, start_time:end_time, :] = chunk
            else:
                # Handle overlap with previous chunk
                overlap_start = max(0, start_time)
                overlap_end = min(start_time + overlap_samples, end_time)
                
                if overlap_start < overlap_end:
                    # Apply crossfade in overlap region
                    overlap_length = overlap_end - overlap_start
                    fade_in = torch.linspace(0, 1, overlap_length).unsqueeze(0).unsqueeze(-1)
                    fade_out = torch.linspace(1, 0, overlap_length).unsqueeze(0).unsqueeze(-1)
                    
                    # Fade out previous audio in overlap region
                    stitched_audio[:, overlap_start:overlap_end, :] *= fade_out
                    # Fade in new audio in overlap region
                    chunk_overlap_start = overlap_start - start_time
                    chunk_overlap_end = chunk_overlap_start + overlap_length
                    stitched_audio[:, overlap_start:overlap_end, :] += chunk[:, chunk_overlap_start:chunk_overlap_end, :] * fade_in
                
                # Add non-overlapping part
                non_overlap_start = overlap_end
                if non_overlap_start < end_time:
                    chunk_non_overlap_start = non_overlap_start - start_time
                    stitched_audio[:, non_overlap_start:end_time, :] = chunk[:, chunk_non_overlap_start:, :]
        
        return stitched_audio


class SpeakerSeparator:
    """Class for separating speakers from audio."""
    
    def __init__(self):
        self.sepformer = None
        self.asteroid_models = {}
    
    def load_speechbrain_model(self):
        """Load SpeechBrain SepformerSeparation model."""
        if SepformerSeparation is None:
            raise ImportError("SpeechBrain SepformerSeparation not available. Please install speechbrain.")
        
        if self.sepformer is None:
            try:
                # Force copy strategy
                os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"
                
                # Load SpeechBrain SepformerSeparation model
                self.sepformer = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-wsj02mix",
                    savedir="pretrained_models/sepformer-wsj02mix"
                )
            except OSError as e:
                if "WinError 1314" in str(e):
                    print("Windows symlink issue detected. Trying alternative approach...")
                    # Try using a different savedir that doesn't require symlinks
                    self.sepformer = SepformerSeparation.from_hparams(
                        source="speechbrain/sepformer-wsj02mix",
                        savedir=None  # Use default cache location
                    )
                else:
                    raise e
        
        return self.sepformer
    
    def load_asteroid_model(self, num_speakers):
        """Load Asteroid model for specified number of speakers."""
        if num_speakers not in self.asteroid_models:
            if num_speakers == 2:
                self.asteroid_models[2] = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
            elif num_speakers == 3:
                self.asteroid_models[3] = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepclean_8k")
            else:
                raise ValueError("Asteroid model only supports 2 or 3 speakers")
        
        return self.asteroid_models[num_speakers]
    
    def separate_audio_speechbrain_sepformer(self, audio_file, out_dir="separated_tracks", chunk_length_seconds=10, overlap_seconds=1):
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
        os.makedirs(out_dir, exist_ok=True)
        
        # Load model if not already loaded
        sepformer = self.load_speechbrain_model()
        
        # Load and resample audio
        wav, sr = torchaudio.load(audio_file)
        if sr != 8000:
            wav = torchaudio.transforms.Resample(sr, 8000)(wav)
        wav = wav.mean(dim=0).unsqueeze(0)  # Convert to mono
        
        print(f"Processing audio with chunking: {chunk_length_seconds}s chunks, {overlap_seconds}s overlap")
        
        # Break audio into chunks
        chunks, start_times = AudioProcessor.chunk_audio(wav, chunk_length_seconds, overlap_seconds, 8000)
        print(f"Created {len(chunks)} chunks for processing")
        
        # Process each chunk separately
        separated_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            est_sources_chunk = sepformer.separate_batch(chunk)  # shape: [1, time, 2]
            separated_chunks.append(est_sources_chunk)
        
        # Stitch the separated chunks back together
        print("Stitching separated chunks back together...")
        overlap_samples = int(overlap_seconds * 8000)
        est_sources = AudioProcessor.stitch_separated_chunks(separated_chunks, start_times, overlap_samples, wav.shape[1])
        
        # Save separated sources
        separated_files = []
        for i in range(est_sources.shape[-1]):
            path = os.path.join(out_dir, f"speaker_sep_{i+1}.wav")
            torchaudio.save(
                path,
                est_sources[0, :, i].unsqueeze(0).cpu(),
                8000
            )
            separated_files.append(path)
        
        print(f"Saved {len(separated_files)} separated audio files to {out_dir}")
        return separated_files
    
    def separate_speakers_asteroid(self, audio_path, num_speakers=2):
        """
        Separate speakers using Asteroid source separation.
        
        Args:
            audio_path (str): Path to the audio file
            num_speakers (int): Number of speakers to separate (2 or 3)
            
        Returns:
            list: List of separated audio arrays for each speaker
        """
        print(f"Separating {num_speakers} speakers using Asteroid...")
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:  # Convert stereo to mono
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Load Asteroid model
        model = self.load_asteroid_model(num_speakers)
        
        # Separate sources
        with torch.no_grad():
            separated_sources = model(audio.unsqueeze(0))  # Add batch dimension
        
        # Convert back to numpy arrays
        separated_audios = []
        for i in range(num_speakers):
            separated_audio = separated_sources[0, i].cpu().numpy()  # Remove batch dimension
            separated_audios.append(separated_audio)
        
        print(f"Successfully separated {num_speakers} speakers")
        return separated_audios, sr


# Convenience functions for backward compatibility
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
    separator = SpeakerSeparator()
    return separator.separate_audio_speechbrain_sepformer(audio_file, out_dir, chunk_length_seconds, overlap_seconds)


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
    separator = SpeakerSeparator()
    return separator.separate_audio_speechbrain_sepformer(audio_file, out_dir, chunk_length_seconds, overlap_seconds)


def separate_speakers_asteroid(audio_path, num_speakers=2):
    """
    Separate speakers using Asteroid source separation.
    
    Args:
        audio_path (str): Path to the audio file
        num_speakers (int): Number of speakers to separate (2 or 3)
        
    Returns:
        list: List of separated audio arrays for each speaker
    """
    separator = SpeakerSeparator()
    return separator.separate_speakers_asteroid(audio_path, num_speakers)


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
    return AudioProcessor.chunk_audio(audio_tensor, chunk_length_seconds, overlap_seconds, sample_rate)


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
    return AudioProcessor.stitch_separated_chunks(separated_chunks_list, start_times, overlap_samples, original_length)


if __name__ == "__main__":
    # Test the separation functionality
    print("Testing speaker separation functionality...")
    
    # Test Asteroid separation
    try:
        separator = SpeakerSeparator()
        print("SpeakerSeparator initialized successfully")
    except Exception as e:
        print(f"Error initializing SpeakerSeparator: {e}")
    
    print("Separation module ready!")
