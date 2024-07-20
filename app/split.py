from pydub import AudioSegment
import os
import logging
from app.utils import ensure_dir, safe_file_operation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@safe_file_operation
def split_audio_by_diarization(audio_path, diarization, output_dir):
    """
    Splits the audio into chunks based on the diarization output.

    Args:
        audio_path (str): Path to the input audio file.
        diarization (pyannote.core.Annotation): Diarization annotation.
        output_dir (str): Directory to save the audio chunks.

    Returns:
        list: List of paths to the generated audio chunks.
    """
    ensure_dir(output_dir)
    audio = AudioSegment.from_wav(audio_path)
    chunk_files = []
    min_chunk_duration_ms = 100  # Minimum duration for chunks in milliseconds

    logging.info(f"Starting to split audio file: {audio_path}")
    
    for i, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start = int(segment.start * 1000)  # milliseconds
        end = int(segment.end * 1000)  # milliseconds

        if end - start < min_chunk_duration_ms:
            logging.warning(
                f"Skipping short chunk: {start}ms to {end}ms for speaker {speaker}"
            )
            continue

        chunk = audio[start:end]
        chunk_path = os.path.join(
            output_dir,
            f"{os.path.basename(audio_path).replace('.wav', '')}_speaker_{speaker}_chunk_{start}_{i}.wav",
        )
        chunk.export(chunk_path, format="wav")
        logging.info(f"Saved audio chunk for Speaker {speaker} to {chunk_path}")
        chunk_files.append(chunk_path)

    logging.info(f"Total number of chunks created: {len(chunk_files)}")
    return chunk_files

if __name__ == "__main__":
    # Example usage and testing
    from pyannote.core import Annotation, Segment
    import numpy as np
    from scipy.io import wavfile
    
    # Create a mock audio file
    def create_mock_audio(path, duration=10, sample_rate=16000):
        t = np.linspace(0, duration, duration * sample_rate, False)
        audio = np.sin(2*np.pi*440*t) * 0.3  # 440 Hz sine wave
        wavfile.write(path, sample_rate, (audio * 32767).astype(np.int16))
    
    # Mock data for testing
    mock_audio_path = "./test/data/test_audio.wav"
    mock_output_dir = "./test/output/chunks"
    
    # Ensure test directories exist
    ensure_dir(os.path.dirname(mock_audio_path))
    ensure_dir(mock_output_dir)
    
    # Create a mock audio file
    create_mock_audio(mock_audio_path)
    
    # Create a mock diarization annotation
    mock_diarization = Annotation()
    mock_diarization[Segment(0, 3)] = "SPEAKER_1"
    mock_diarization[Segment(3, 6)] = "SPEAKER_2"
    mock_diarization[Segment(6, 10)] = "SPEAKER_1"

    try:
        # Test split_audio_by_diarization
        chunk_files = split_audio_by_diarization(
            mock_audio_path,
            mock_diarization,
            mock_output_dir
        )
        print(f"Created {len(chunk_files)} audio chunks:")
        for chunk in chunk_files:
            print(f"  - {chunk}")

    except Exception as e:
        logging.error(f"Error during audio splitting test: {str(e)}")