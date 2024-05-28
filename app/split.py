from pydub import AudioSegment
import os
import logging
from app.utils import ensure_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    for i, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start = int(segment.start * 1000)  # milliseconds
        end = int(segment.end * 1000)  # milliseconds
        chunk = audio[start:end]
        chunk_path = os.path.join(output_dir, f"{os.path.basename(audio_path).replace('.wav', '')}_speaker_{speaker}_chunk_{start}_{i}.wav")
        chunk.export(chunk_path, format="wav")
        logging.info(f"Saved audio chunk for Speaker {speaker} to {chunk_path}")
        chunk_files.append(chunk_path)
    return chunk_files