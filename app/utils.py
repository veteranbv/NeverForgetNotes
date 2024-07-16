import os
import re
from datetime import datetime
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def ensure_dir(directory):
    """
    Ensures the specified directory exists, creating it if necessary.

    Args:
        directory (str): The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory {directory}")


def format_seconds(seconds):
    """
    Formats a duration in seconds into a string in the format HH:MM:SS.

    Args:
        seconds (int): The duration in seconds.

    Returns:
        str: The formatted time string.
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def sanitize_filename(name):
    """
    Sanitizes a string to be used as a filename by replacing invalid characters.

    Args:
        name (str): The original filename.

    Returns:
        str: The sanitized filename.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).lower()


def prompt_for_recording_details(default_date, default_name, filename):
    """
    Prompts the user for custom recording details, with defaults provided.

    Args:
        default_date (str): The default recording date.
        default_name (str): The default recording name.
        filename (str): The name of the audio file.

    Returns:
        tuple: The recording date, sanitized name, and OpenAI usage flag.
    """
    use_custom_details = (
        input(
            f"Do you want to provide a custom recording date and meeting name for {filename}? (y/n): "
        )
        .strip()
        .lower()
    )
    if use_custom_details == "y":
        recording_date = input("Enter the recording date (YYYY-MM-DD): ")
        recording_name = input("Enter the name for the recording: ")
        try:
            normalized_date = datetime.strptime(recording_date, "%Y-%m-%d").strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            logging.error("Invalid date format. Please use YYYY-MM-DD.")
            raise
    else:
        normalized_date = default_date
        recording_name = default_name
    sanitized_name = sanitize_filename(recording_name)

    use_openai = (
        input("Would you like to use OpenAI for transcription? (y/n): ").strip().lower()
        == "y"
    )

    return normalized_date, sanitized_name, use_openai


def list_available_prompts(prompts_dir):
    """
    Lists all available prompt files in the specified directory.

    Args:
        prompts_dir (str): Directory containing prompt files.

    Returns:
        list: A list of available prompt file paths.
    """
    return [
        os.path.join(prompts_dir, file)
        for file in os.listdir(prompts_dir)
        if file.endswith(".txt")
    ]


def select_prompt(prompts):
    """
    Prompts the user to select a prompt from the available list.

    Args:
        prompts (list): List of available prompt file paths.

    Returns:
        str: The path to the selected prompt file.
    """
    print("Which prompt would you like to use for this run?")
    for idx, prompt in enumerate(prompts):
        print(f"{idx + 1}) {os.path.basename(prompt)}")

    while True:
        try:
            choice = int(input("Enter the number of the prompt: ")) - 1
            if 0 <= choice < len(prompts):
                return prompts[choice]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def create_output_dirs(base_output_dir, normalized_date, recording_name):
    """
    Creates necessary output directories for processing and storing results.

    Args:
        base_output_dir (str): The base directory for outputs.
        normalized_date (str): The normalized date string.
        recording_name (str): The sanitized recording name.

    Returns:
        dict: A dictionary with paths to the created directories.
    """
    output_dir = os.path.join(base_output_dir, f"{normalized_date}-{recording_name}")
    dirs = {
        "transcriptions": os.path.join(output_dir, "transcriptions"),
        "diarizations": os.path.join(output_dir, "diarizations"),
        "merged_output": os.path.join(output_dir, "merged_output"),
        "summary": os.path.join(output_dir, "summary"),
        "figures": os.path.join(output_dir, "figures"),
        "temp_wav": "./temp/wav_files",
        "temp_chunks": "./temp/chunks",
        "temp_transcriptions": "./temp/transcriptions",
    }
    for directory in dirs.values():
        ensure_dir(directory)
    return dirs


def get_audio_metadata(file_path):
    """
    Retrieves the recording date from the audio file's metadata using ffprobe.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        str: The recording date in YYYY-MM-DD format, or today's date if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=creation_time",
                "-of",
                "default=nw=1:nk=1",
                file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        date = result.stdout.strip()
        if date:
            return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
    except (subprocess.CalledProcessError, ValueError) as e:
        logging.warning(
            f"Error retrieving metadata for {file_path}: {str(e)}. Trying file system dates."
        )

    # Fallback to file system creation or modification date
    try:
        stat = os.stat(file_path)
        creation_time = (
            stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime
        )
        return datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d")
    except Exception as e:
        logging.warning(
            f"Error retrieving file system date for {file_path}: {str(e)}. Using current date."
        )
        return datetime.now().strftime("%Y-%m-%d")
