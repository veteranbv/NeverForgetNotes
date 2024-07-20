import os
import logging
import shutil
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from app.audio_processing import process_files
from app.utils import (
    list_available_prompts,
    select_prompt,
    is_audio_file,
    is_transcript_file,
    ensure_dir,
    cleanup_old_logs,
    validate_environment_variables,
    format_time,
    get_file_size,
)
from app.audio_utils import get_audio_length
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.prompt import Confirm, Prompt

console = Console()

def setup_logging():
    """
    Set up logging configuration for the application.
    """
    logs_dir = "./logs"
    ensure_dir(logs_dir)
    log_file = os.path.join(logs_dir, "application.log")

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add only the file handler
    root_logger.addHandler(file_handler)

def prompt_for_file_settings(
    filename, available_prompts, openai_model, anthropic_model
):
    """
    Prompt the user for settings for a specific file.
    """
    console.print(f"\n[bold blue]Settings for file: {filename}[/bold blue]")
    
    # 1. Transcription method
    use_openai = Confirm.ask("Would you like to use OpenAI for transcription?")

    # 2. Summarization model
    print("Choose summarization model:")
    print(f"1) {openai_model}")
    print(f"2) {anthropic_model}")
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            summarization_model = openai_model if choice == "1" else anthropic_model
            break
        print("Invalid choice. Please enter 1 or 2.")

    # 3. Prompt selection
    prompt = select_prompt(available_prompts)

    # 4. Custom recording details
    use_custom_details = Confirm.ask("Do you want to provide custom recording details?")
    if use_custom_details:
        recording_date = Prompt.ask("Enter the recording date (YYYY-MM-DD)")
        recording_name = Prompt.ask("Enter the name for the recording")
    else:
        recording_date = None
        recording_name = None

    return use_openai, summarization_model, prompt, recording_date, recording_name

def prompt_for_global_settings(available_prompts):
    """
    Prompt the user for global settings to apply to all files.
    """
    # 1. Transcription method
    use_openai = Confirm.ask("Would you like to use OpenAI for transcription?")

    # 2. Summarization model
    openai_model = os.getenv("OPENAI_MODEL")
    anthropic_model = os.getenv("ANTHROPIC_MODEL")
    summarization_model_choice = get_user_choice(
        "Choose summarization model:", [openai_model, anthropic_model]
    )
    summarization_model = openai_model if summarization_model_choice == 1 else anthropic_model

    # 3. Prompt selection
    use_different_prompts = Confirm.ask("Do you want to choose different prompts for each input file?")
    global_prompt = None if use_different_prompts else select_prompt(available_prompts)

    return use_openai, summarization_model, use_different_prompts, global_prompt

def get_user_choice(prompt, options):
    """
    Custom function to get user input for choices.
    """
    while True:
        print(prompt)
        for i, option in enumerate(options, 1):
            print(f"{i}) {option}")
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            return int(choice)
        print("Invalid choice. Please enter 1 or 2.")

def cleanup_temp_directories():
    """
    Clean up temporary directories created during processing.
    """
    temp_dir = "./temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")

def main():
    """
    Main function to initialize and start the file processing pipeline.
    """
    console.print(
        Panel.fit("Starting audio processing application...", border_style="bold blue")
    )

    setup_logging()
    logging.info("Starting audio processing application...")

    # Load environment variables
    load_dotenv()

    # Clean up old log files
    cleanup_old_logs("./logs")

    # Define directories
    audio_dir = "./audio"
    input_dir = os.path.join(audio_dir, "input")
    not_processed_dir = os.path.join(audio_dir, "not_processed")
    processed_dir = os.path.join(audio_dir, "processed")
    base_output_dir = "./output"

    # Ensure directories exist
    for directory in [
        audio_dir,
        input_dir,
        not_processed_dir,
        processed_dir,
        base_output_dir,
    ]:
        ensure_dir(directory)

    # Fetch and validate environment variables
    required_vars = [
        "HF_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_MODEL_TOKEN_LIMIT",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_MODEL_TOKEN_LIMIT",
    ]
    if not validate_environment_variables(required_vars):
        console.print(
            "Error: Missing required environment variables. Please check your .env file.",
            style="bold red",
        )
        return

    # List available prompts
    prompts_dir = "./prompts/library"
    ensure_dir(prompts_dir)
    available_prompts = list_available_prompts(prompts_dir)

    if not available_prompts:
        console.print(
            f"No prompt files found in {prompts_dir}. Please add some prompt files.",
            style="bold red",
        )
        return

    # Get list of files to process
    input_files = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
    ]
    audio_files = [f for f in input_files if is_audio_file(f)]
    transcript_files = [f for f in input_files if is_transcript_file(f)]

    if not audio_files and not transcript_files:
        console.print(
            f"No audio or transcript files found in {input_dir}. Please add some files to process.",
            style="bold red",
        )
        return

    console.print(
        f"Found {len(audio_files)} audio files and {len(transcript_files)} transcript files."
    )

    # Prompt for global settings
    use_global_settings = Confirm.ask("Do you want to have the same settings for all of the input files?")

    if use_global_settings:
        global_use_openai, global_summarization_model, global_use_different_prompts, global_prompt = prompt_for_global_settings(available_prompts)
        console.print(
            f"Chosen summarization model: {global_summarization_model}",
            style="bold green",
        )
    else:
        global_use_openai = None
        global_summarization_model = None
        global_use_different_prompts = True
        global_prompt = None

    # Prepare file settings
    file_settings = []
    for filename in audio_files + transcript_files:
        if use_global_settings:
            file_settings.append(
                {
                    "filename": filename,
                    "use_openai": global_use_openai,
                    "summarization_model": global_summarization_model,
                    "prompt": global_prompt if not global_use_different_prompts else select_prompt(available_prompts),
                    "recording_date": None,
                    "recording_name": None,
                    "length": get_audio_length(os.path.join(input_dir, filename)),
                    "size": get_file_size(os.path.join(input_dir, filename)),
                }
            )
        else:
            use_openai, summarization_model, prompt, recording_date, recording_name = prompt_for_file_settings(
                filename,
                available_prompts,
                os.getenv("OPENAI_MODEL"),
                os.getenv("ANTHROPIC_MODEL"),
            )
            file_settings.append(
                {
                    "filename": filename,
                    "use_openai": use_openai,
                    "summarization_model": summarization_model,
                    "prompt": prompt,
                    "recording_date": recording_date,
                    "recording_name": recording_name,
                    "length": get_audio_length(os.path.join(input_dir, filename)),
                    "size": get_file_size(os.path.join(input_dir, filename)),
                }
            )

    # Display job overview
    console.print(Panel("Job Overview", border_style="bold green"))
    file_table = Table(show_header=True, header_style="bold magenta")
    file_table.add_column("File Name", style="dim")
    file_table.add_column("Length")
    file_table.add_column("Size")
    file_table.add_column("Transcription")
    file_table.add_column("Summarization Model")
    file_table.add_column("Prompt")
    file_table.add_column("Custom Details")

    for settings in file_settings:
        file_table.add_row(
            settings["filename"],
            format_time(settings["length"]),
            f"{settings['size']:.2f} MB",
            "OpenAI" if settings["use_openai"] else "Whisper",
            settings["summarization_model"],
            os.path.basename(settings["prompt"]) if settings["prompt"] else "Default",
            "Yes" if settings["recording_date"] or settings["recording_name"] else "No",
        )
    console.print(file_table)

    # Process files
    successfully_processed_files = 0
    total_processing_time = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        overall_task = progress.add_task(
            "[bold blue]Overall Progress", total=len(file_settings)
        )

        for settings in file_settings:
            file_task = progress.add_task(
                f"Processing: {settings['filename']}", total=100
            )

            try:
                file_processing_time = process_files(
                    input_dir,
                    processed_dir,
                    base_output_dir,
                    os.getenv("HF_AUTH_TOKEN"),
                    os.getenv("OPENAI_API_KEY"),
                    os.getenv("ANTHROPIC_API_KEY"),
                    settings["prompt"],
                    settings["summarization_model"],
                    [settings["filename"]]
                    if is_audio_file(settings["filename"])
                    else [],
                    [settings["filename"]]
                    if is_transcript_file(settings["filename"])
                    else [],
                    os.getenv("OPENAI_MODEL"),
                    os.getenv("ANTHROPIC_MODEL"),
                    int(os.getenv("OPENAI_MODEL_TOKEN_LIMIT")),
                    int(os.getenv("ANTHROPIC_MODEL_TOKEN_LIMIT")),
                    use_openai=settings["use_openai"],
                    recording_date=settings["recording_date"],
                    recording_name=settings["recording_name"],
                    progress=progress,
                    file_task=file_task,
                )
                successfully_processed_files += 1
                total_processing_time += file_processing_time
            except Exception as e:
                logging.exception(f"An error occurred during file processing: {str(e)}")
                console.print(
                    f"Error processing {settings['filename']}: {str(e)}",
                    style="bold red",
                )

            progress.update(overall_task, advance=1)

    # Clean up temporary directories
    cleanup_temp_directories()

    console.print(
        Panel("Audio processing application completed.", border_style="bold green")
    )

    # Display summary information
    console.print(f"Total files processed successfully: {successfully_processed_files}")
    console.print(f"Total processing time: {format_time(total_processing_time)}")

if __name__ == "__main__":
    main()