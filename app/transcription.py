import whisper
import torch
import os
import logging
import openai

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def transcribe_audio(audio_path, output_dir, openai_api_key):
    """
    Transcribes a single audio chunk using Whisper with GPU if available.

    Args:
        audio_path (str): Path to the audio chunk.
        output_dir (str): Directory to save the transcription.
        openai_api_key (str): OpenAI API key.

    Returns:
        str: The transcribed text.
    """
    try:
        logging.info(f"Starting transcription for {audio_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("small", device=device)
        result = model.transcribe(audio_path)
        full_transcription = result["text"]

        output_file = os.path.join(
            output_dir,
            os.path.basename(audio_path).replace(".wav", "_transcription.txt"),
        )
        ensure_dir(output_dir)
        with open(output_file, "w") as f:
            f.write(full_transcription)

        logging.info(f"Transcription saved for {audio_path} to {output_file}")
        return full_transcription

    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {audio_path} - {str(fnf_error)}")
        raise
    except whisper.WhisperError as whisper_error:
        logging.error(f"Whisper model error for {audio_path}: {str(whisper_error)}")
        raise
    except Exception as e:
        logging.error(f"Error in transcription for {audio_path}: {str(e)}")
        raise


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
        openai.api_key = openai_api_key
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
        with open(output_file, "w") as f:
            f.write(full_transcription)

        logging.info(f"OpenAI transcription saved for {audio_path} to {output_file}")
        return full_transcription

    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {audio_path} - {str(fnf_error)}")
        raise
    except openai.OpenAIError as openai_error:
        logging.error(f"OpenAI API error for {audio_path}: {str(openai_error)}")
        raise
    except Exception as e:
        logging.error(f"Error in OpenAI transcription for {audio_path}: {str(e)}")
        raise


def ensure_dir(directory):
    """
    Ensures the specified directory exists, creating it if necessary.

    Args:
        directory (str): The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory {directory}")
