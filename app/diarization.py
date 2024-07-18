from pyannote.audio import Pipeline
import os
import logging
from app.utils import sanitize_filename, safe_file_operation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@safe_file_operation
def diarize_audio(audio_path, output_dir, hf_auth_token, progress=None):
    """
    Performs speaker diarization on an audio file using Pyannote.

    Args:
        audio_path (str): Path to the input audio file.
        output_dir (str): Directory to save the diarization output.
        hf_auth_token (str): Hugging Face authentication token.
        progress (callable, optional): A callback function to update progress.

    Returns:
        pyannote.core.Annotation: The diarization annotation.
    """
    try:
        logging.info(f"Initializing diarization pipeline for {audio_path}")
        # Initialize the diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_auth_token
        )
        logging.info("Diarization pipeline initialized. Starting diarization...")
        
        if progress:
            progress(0)  # Signal start of diarization

        # Perform diarization
        diarization = pipeline(audio_path)
        
        if progress:
            progress(50)  # Signal halfway point of diarization

        logging.info("Diarization completed. Saving output...")

        # Create a sanitized filename for the RTTM output
        base_name = os.path.basename(audio_path)
        sanitized_name = sanitize_filename(os.path.splitext(base_name)[0])
        output_file = os.path.join(output_dir, f"{sanitized_name}_diarization.rttm")

        # Save the diarization output to a file
        with open(output_file, "w") as f:
            diarization.write_rttm(f)

        logging.info(f"Diarization saved for {audio_path} to {output_file}")
        
        if progress:
            progress(100)  # Signal completion of diarization

        return diarization
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {audio_path} - {str(fnf_error)}")
        raise
    except Exception as e:
        logging.error(f"Error in diarization for {audio_path}: {str(e)}")
        raise

def get_speaker_segments(diarization):
    """
    Extracts speaker segments from the diarization output.

    Args:
        diarization (pyannote.core.Annotation): The diarization annotation.

    Returns:
        list: A list of tuples containing (start_time, end_time, speaker_label).
    """
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append((turn.start, turn.end, speaker))
    return speaker_segments

if __name__ == "__main__":
    # Example usage and testing
    from dotenv import load_dotenv
    load_dotenv()

    test_audio_path = "./test/data/test_audio.wav"
    test_output_dir = "./test/output"
    hf_auth_token = os.getenv("HF_AUTH_TOKEN")

    def print_progress(progress):
        print(f"Diarization progress: {progress}%")

    try:
        # Perform diarization
        diarization_result = diarize_audio(test_audio_path, test_output_dir, hf_auth_token, print_progress)

        # Extract and print speaker segments
        speaker_segments = get_speaker_segments(diarization_result)
        for start, end, speaker in speaker_segments:
            print(f"Speaker {speaker}: {start:.2f}s - {end:.2f}s")

    except Exception as e:
        logging.error(f"Error during diarization test: {str(e)}")