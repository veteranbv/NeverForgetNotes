import os
import logging
from app.audio_processing import process_audio_files
from app.utils import list_available_prompts, select_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main function to initialize and start the audio processing pipeline.
    """
    # Define directories
    input_dir = "./audio/input"
    processed_dir = "./audio/processed"
    base_output_dir = "./output"
    prompts_dir = "./prompts/library"

    # Fetch environment variables
    hf_auth_token = os.getenv("HF_AUTH_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if environment variables are set
    if not hf_auth_token:
        logging.error("HF_AUTH_TOKEN is not set. Please set the environment variable.")
        return
    if not openai_api_key:
        logging.error("OPENAI_API_KEY is not set. Please set the environment variable.")
        return

    # List available prompts and prompt user for selection
    available_prompts = list_available_prompts(prompts_dir)
    selected_prompt_file = select_prompt(available_prompts)

    # Process audio files
    try:
        process_audio_files(
            input_dir,
            processed_dir,
            base_output_dir,
            hf_auth_token,
            openai_api_key,
            selected_prompt_file,
        )
    except Exception as e:
        logging.error(f"An error occurred during audio processing: {str(e)}")


if __name__ == "__main__":
    main()
