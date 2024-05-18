import whisper
import os
import logging

def transcribe_audio(audio_path, output_dir, openapi_token):
    """Transcribes a single audio chunk using Whisper.

    Args:
        audio_path (str): Path to the audio chunk.
        output_dir (str): Directory to save the transcription.
        openapi_token (str): OpenAI API token.

    Returns:
        str: The transcribed text.
    """
    try:
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        full_transcription = result['text']

        output_file = os.path.join(output_dir, os.path.basename(audio_path).replace('.wav', '_transcription.txt'))
        with open(output_file, 'w') as f:
            f.write(full_transcription)

        # logging.info(f"Transcription saved for {audio_path}") 
        return full_transcription

    except Exception as e:
        logging.error(f"Error in transcription for {audio_path}: {e}")
        return None