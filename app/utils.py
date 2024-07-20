import os
import uuid
import time
from datetime import datetime, timedelta
import subprocess
import logging
from slugify import slugify
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ensure_dir(directory):
    """
    Ensures the specified directory exists, creating it if necessary.

    Args:
        directory (str): The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def cleanup_old_logs(logs_dir, max_age_days=30):
    """
    Removes log files older than the specified number of days.

    Args:
        logs_dir (str): The directory containing log files.
        max_age_days (int): The maximum age of log files in days.
    """
    current_time = time.time()
    for filename in os.listdir(logs_dir):
        file_path = os.path.join(logs_dir, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_days * 86400:  # 86400 seconds in a day
                os.remove(file_path)
                logging.info(f"Removed old log file: {filename}")

def format_seconds(seconds):
    """
    Formats a duration in seconds into a string in the format HH:MM:SS.

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: The formatted time string.
    """
    return str(timedelta(seconds=int(seconds)))

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds.
    
    Returns:
        str: Formatted time string.
    """
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)

def get_file_size(file_path):
    """
    Get the size of a file in megabytes.

    Args:
        file_path (str): Path to the file.

    Returns:
        float: Size of the file in megabytes.
    """
    return os.path.getsize(file_path) / (1024 * 1024)

def sanitize_filename(name, max_length=242):
    """
    Sanitizes a string to be used as a filename, ensuring cross-platform compatibility.

    Args:
        name (str): The original filename.
        max_length (int): Maximum length of the filename (default is 242 for maximum compatibility).

    Returns:
        str: The sanitized filename.
    """
    safe_name = slugify(name, lowercase=True, regex_pattern=r"[^\w.-]")
    safe_name = safe_name.strip(".")
    if not safe_name:
        safe_name = "unnamed_file"
    root, ext = os.path.splitext(safe_name)
    max_root_length = max_length - len(ext) - 1
    if len(root) > max_root_length:
        root = root[:max_root_length]
    return f"{root}{ext}"

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
    print("Available prompts:")
    for idx, prompt in enumerate(prompts, 1):
        print(f"{idx}) {os.path.basename(prompt)}")

    while True:
        try:
            choice = int(input("Enter the number of the prompt: ")) - 1
            if 0 <= choice < len(prompts):
                return prompts[choice]
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
    unique_id = str(uuid.uuid4())[:8]
    dirs = {
        "transcriptions": os.path.join(output_dir, "transcriptions"),
        "diarizations": os.path.join(output_dir, "diarizations"),
        "merged_output": os.path.join(output_dir, "merged_output"),
        "summary": os.path.join(output_dir, "summary"),
        "figures": os.path.join(output_dir, "figures"),
        "temp_wav": os.path.join("./temp", unique_id, "wav_files"),
        "temp_chunks": os.path.join("./temp", unique_id, "chunks"),
        "temp_transcriptions": os.path.join("./temp", unique_id, "transcriptions"),
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
                "-v", "error",
                "-show_entries", "format_tags=creation_time",
                "-of", "default=nw=1:nk=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        date = result.stdout.strip()
        if date:
            return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
    except (subprocess.CalledProcessError, ValueError) as e:
        logging.warning(f"Error retrieving metadata for {file_path}: {str(e)}. Trying file system dates.")

    try:
        stat = os.stat(file_path)
        creation_time = stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime
        return datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d")
    except Exception as e:
        logging.warning(f"Error retrieving file system date for {file_path}: {str(e)}. Using current date.")
        return datetime.now().strftime("%Y-%m-%d")

def is_audio_file(file_path):
    """
    Checks if the given file is an audio file based on its extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file is an audio file, False otherwise.
    """
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
    return os.path.splitext(file_path)[1].lower() in audio_extensions

def is_transcript_file(file_path):
    """
    Checks if the given file is a transcript file (.txt or .vtt).

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file is a transcript, False otherwise.
    """
    return file_path.lower().endswith((".txt", ".vtt"))

def list_directory_contents(directory):
    """
    Logs the contents of a directory.

    Args:
        directory (str): The directory to list the contents of.
    """
    logging.info(f"Contents of {directory}:")
    for root, dirs, files in os.walk(directory):
        for name in files:
            logging.info(f"  - {os.path.join(root, name)}")
        for name in dirs:
            logging.info(f"  - {os.path.join(root, name)}/")

def estimate_tokens(text, model="gpt-4"):
    """
    Estimates the number of tokens in the given text.

    Args:
        text (str): The text to estimate tokens for.
        model (str): The model to use for estimation (default is "gpt-4").

    Returns:
        int: Estimated number of tokens.
    """
    # This is a very rough estimate. For more accurate results, consider using tiktoken library.
    return int(len(text.split()) * 1.3)

def chunk_text(text, max_tokens):
    """
    Splits a long text into chunks that fit within the token limit.

    Args:
        text (str): The full text to chunk.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        word_token_count = estimate_tokens(word)
        if current_token_count + word_token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_token_count = 0
        current_chunk.append(word)
        current_token_count += word_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def safe_file_operation(func):
    """
    Decorator to ensure safe file operations with proper error handling.

    Args:
        func: The function to wrap.

    Returns:
        function: The wrapped function with error handling.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IOError as e:
            logging.error(f"IO error occurred during file operation: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during file operation: {str(e)}")
            raise
    return wrapper

@safe_file_operation
def read_file(file_path):
    """
    Safely reads a file and returns its content.

    Args:
        file_path (str): Path to the file to read.

    Returns:
        str: Content of the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

@safe_file_operation
def write_file(file_path, content):
    """
    Safely writes content to a file.

    Args:
        file_path (str): Path to the file to write.
        content (str): Content to write to the file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

def validate_environment_variables(required_vars):
    """
    Validates that all required environment variables are set.

    Args:
        required_vars (list): List of required environment variable names.

    Returns:
        bool: True if all variables are set, False otherwise.
    """
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(f"The following required environment variables are not set: {', '.join(missing_vars)}")
        return False
    return True

if __name__ == "__main__":
    # Example usage and testing
    test_dir = "./test_utils"
    ensure_dir(test_dir)

    # Test sanitize_filename
    print(sanitize_filename("Test File Name! @#$%.txt"))

    # Test format_time
    print(format_time(3661))  # Should print "1h 1m 1s"

    # Test get_file_size
    test_file_path = os.path.join(test_dir, "test_file.txt")
    with open(test_file_path, "w") as f:
        f.write("Test content" * 1000)
    print(f"File size: {get_file_size(test_file_path):.2f} MB")

    # Clean up
    shutil.rmtree(test_dir)