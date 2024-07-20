import whisper
import torch
import os
import logging
import openai
from app.utils import ensure_dir, safe_file_operation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@safe_file_operation
def transcribe_audio(audio_path, output_dir, openai_api_key):
    """
    Transcribes a single audio chunk using Whisper with GPU if available.

    Args:
        audio_path (str): Path to the audio chunk.
        output_dir (str): Directory to save the transcription.
        openai_api_key (str): OpenAI API key (not used in this function, but kept for consistency).

    Returns:
        str: The transcribed text.
    """
    try:
        logging.info(f"Starting transcription for {audio_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", device=device)
        result = model.transcribe(audio_path)
        full_transcription = result["text"]

        output_file = os.path.join(
            output_dir,
            os.path.basename(audio_path).replace(".wav", "_transcription.txt"),
        )
        ensure_dir(output_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_transcription)

        logging.info(f"Transcription saved for {audio_path} to {output_file}")
        return full_transcription

    except Exception as e:
        logging.error(f"Error in transcription for {audio_path}: {str(e)}")
        raise

@safe_file_operation
def transcribe_audio_with_openai(audio_path, output_dir, openai_api_key):
    """
    Transcribes an audio file using OpenAI's API.

    Args:
        audio_path (str): Path to the audio file.
        output_dir (str): Directory to save the transcription.
        openai_api_key (str): OpenAI API key.

    Returns:
        str: The transcribed text.
    """
    try:
        logging.info(f"Starting OpenAI transcription for {audio_path}")
        client = openai.OpenAI(api_key=openai_api_key)

        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )

        full_transcription = transcription.text

        output_file = os.path.join(
            output_dir,
            os.path.basename(audio_path).replace(".wav", "_transcription.txt"),
        )
        ensure_dir(output_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_transcription)

        logging.info(f"OpenAI transcription saved for {audio_path} to {output_file}")
        return full_transcription

    except Exception as e:
        logging.error(f"Error in OpenAI transcription for {audio_path}: {str(e)}")
        raise

def transcribe_chunk(chunk_path, output_dir, openai_api_key, use_openai):
    """
    Transcribes a single audio chunk using either local Whisper or OpenAI's API.

    Args:
        chunk_path (str): Path to the audio chunk.
        output_dir (str): Directory to save the transcription.
        openai_api_key (str): OpenAI API key.
        use_openai (bool): Whether to use OpenAI for transcription.

    Returns:
        str: The transcribed text.
    """
    if use_openai:
        return transcribe_audio_with_openai(chunk_path, output_dir, openai_api_key)
    else:
        return transcribe_audio(chunk_path, output_dir, openai_api_key)

if __name__ == "__main__":
    # Example usage and testing
    from dotenv import load_dotenv
    load_dotenv()

    test_audio_path = "./test/data/test_audio.wav"
    test_output_dir = "./test/output"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    try:
        # Test local Whisper transcription
        local_transcription = transcribe_chunk(test_audio_path, test_output_dir, openai_api_key, use_openai=False)
        print("Local Whisper Transcription:", local_transcription)

        # Test OpenAI transcription
        openai_transcription = transcribe_chunk(test_audio_path, test_output_dir, openai_api_key, use_openai=True)
        print("OpenAI Transcription:", openai_transcription)

    except Exception as e:
        logging.error(f"Error during transcription test: {str(e)}")