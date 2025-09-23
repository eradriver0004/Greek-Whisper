import os
import tempfile
import librosa
import soundfile as sf
from pydub import AudioSegment
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"

from pyannote.audio import Pipeline

def diarize(audio_path):
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_path (str): Path to the audio file to diarize
        
    Returns:
        dict: Diarization data in template.json format with tiers for each speaker
    """
    # You need a Hugging Face token (free, from https://huggingface.co/settings/tokens)
    # Make sure to `huggingface-cli login` first or pass use_auth_token="YOUR_TOKEN"
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=True)

    # Check if the audio file can be loaded directly by soundfile
    try:
        # Try to load with soundfile first
        sf.read(audio_path)
        # If successful, use the original file
        processed_audio_path = audio_path
    except Exception as e:
        print(f"Audio file format not supported by soundfile: {e}")
        print("Converting audio file to compatible format...")
        
        # Load with librosa (more permissive) and save as proper WAV
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Create a temporary file with proper WAV format
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)  # Close the file descriptor
            
            # Save as proper WAV file
            sf.write(temp_path, audio, sr, format='WAV', subtype='PCM_16')
            processed_audio_path = temp_path
            print(f"Audio converted and saved to temporary file: {temp_path}")
            
        except Exception as e2:
            print(f"Failed to convert audio with librosa/soundfile: {e2}")
            # Fallback to ffmpeg via pydub (requires ffmpeg installed)
            try:
                temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
                os.close(temp_fd)
                segment = AudioSegment.from_file(audio_path)
                segment = segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16kHz mono 16-bit
                segment.export(temp_path, format='wav')
                processed_audio_path = temp_path
                print(f"Audio converted via ffmpeg/pydub to: {temp_path}")
            except Exception as e3:
                print(f"Failed to convert audio file via ffmpeg/pydub: {e3}")
                raise e3

    try:
        # Apply diarization on the audio file
        diarization = pipeline(processed_audio_path)
    finally:
        # Clean up temporary file if we created one
        if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
            os.unlink(processed_audio_path)

    # Collect diarization results grouped by speaker
    speaker_data = {}
    annotation_id = 1
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_data:
            speaker_data[speaker] = []
        
        speaker_data[speaker].append({
            "id": f"a{annotation_id}",
            "start_time": str(int(turn.start * 1000)),  # Convert to milliseconds as string
            "end_time": str(int(turn.end * 1000)),      # Convert to milliseconds as string
            "value": ""  # Empty value for diarization segments
        })
        annotation_id += 1
    
    # Create template.json structure
    template_data = {
        "audio_url": f"file:///{audio_path.replace(os.sep, '/')}",
        "last_used_annotation_id": annotation_id - 1,
        "tiers": [],
    }
    
    # Add tiers for each speaker using speaker labels as object keys
    for speaker, annotations in speaker_data.items():
        # Sort annotations by start time
        annotations.sort(key=lambda x: int(x["start_time"]))
        
        # Use speaker label as both the object key and tier ID
        template_data[speaker] = {
            "id": speaker,
            "annotations": annotations
        }
        template_data["tiers"].append(speaker)
    
    return template_data

if __name__ == "__main__":
    import sys
    import os
    # Add the parent directory to the path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    
    # Import write_json function
    from utils.utils import write_eaf, write_json
    data = diarize("data/test1.wav")
    write_json(data, "data/test1.json")
    write_eaf(data, "data/test1.eaf")