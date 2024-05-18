from pydub import AudioSegment
import os
import logging

def split_audio_by_diarization(audio_path, diarization, output_dir):
    """Splits the audio into chunks based on the diarization output.
    """
    audio = AudioSegment.from_wav(audio_path)
    for i, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start = int(segment.start * 1000) # milliseconds
        end = int(segment.end * 1000) # milliseconds
        chunk = audio[start:end]
        chunk_path = os.path.join(output_dir, f"{os.path.basename(audio_path).replace('.wav', '')}_speaker_{speaker}_chunk_{start}_{i}.wav") 
        chunk.export(chunk_path, format="wav")
        logging.info(f"Saved audio chunk for Speaker {speaker} to {chunk_path}")  # This message will now be at WARNING level