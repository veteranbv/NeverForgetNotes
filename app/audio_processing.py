import logging
import os
import time
import shutil

from app.utils import (
    ensure_dir,
    format_seconds,
    create_output_dirs,
    get_audio_metadata,
    is_audio_file,
    is_transcript_file,
    list_directory_contents,
    sanitize_filename,
    safe_file_operation,
    read_file,
    write_file,
    chunk_text,
    estimate_tokens,
)
from app.audio_utils import convert_to_wav, get_audio_length, plot_waveform
from app.transcription import transcribe_audio, transcribe_audio_with_openai
from app.diarization import diarize_audio
from app.merge import merge_transcriptions, merge_raw_transcriptions
from app.split import split_audio_by_diarization
from app.summarize import summarize_transcript


@safe_file_operation
def process_audio_file(
    file_path,
    output_dirs,
    hf_auth_token,
    openai_api_key,
    use_openai,
    progress,
    file_task,
):
    """
    Process a single audio file.

    Args:
        file_path (str): Path to the audio file.
        output_dirs (dict): Dictionary of output directories.
        hf_auth_token (str): HuggingFace authentication token.
        openai_api_key (str): OpenAI API key.
        use_openai (bool): Whether to use OpenAI for transcription.
        progress (rich.progress.Progress): Progress bar object.
        file_task (int): Task ID for the file being processed.

    Returns:
        str: Path to the merged output file.
    """
    logging.info(f"Processing audio file: {file_path}")
    step_start_time = time.time()

    sanitized_basename = sanitize_filename(
        os.path.splitext(os.path.basename(file_path))[0]
    )
    wav_file = os.path.join(
        output_dirs["temp_wav"],
        f"{sanitized_basename}.wav",
    )

    try:
        # Convert to WAV
        progress.update(file_task, description="Converting to WAV", completed=10)
        logging.info(f"Converting {file_path} to WAV...")
        convert_to_wav(file_path, wav_file)
        convert_time = time.time() - step_start_time
        logging.info(f"Conversion time: {format_seconds(convert_time)}")

        # Diarization
        progress.update(file_task, description="Diarizing audio", completed=30)
        logging.info(f"Starting diarization for {wav_file}...")
        step_start_time = time.time()
        diarization = diarize_audio(
            wav_file, output_dirs["diarizations"], hf_auth_token
        )
        diarization_time = time.time() - step_start_time
        logging.info(f"Diarization time: {format_seconds(diarization_time)}")

        if diarization is None:
            raise ValueError(f"Diarization failed for {file_path}")

        # Split Audio by Diarization
        progress.update(file_task, description="Splitting audio", completed=40)
        logging.info(f"Splitting audio for {wav_file}...")
        step_start_time = time.time()
        chunk_files = split_audio_by_diarization(
            wav_file, diarization, output_dirs["temp_chunks"]
        )
        split_time = time.time() - step_start_time
        logging.info(f"Split time: {format_seconds(split_time)}")

        if not chunk_files:
            raise ValueError(f"No audio chunks found in {output_dirs['temp_chunks']}")

        # Log the contents of the temp_chunks directory
        list_directory_contents(output_dirs["temp_chunks"])

        # Transcription
        progress.update(file_task, description="Transcribing audio", completed=60)
        logging.info(f"Starting transcription for {len(chunk_files)} chunks...")
        step_start_time = time.time()
        transcriptions = []
        for chunk_file in chunk_files:
            if use_openai:
                transcription = transcribe_audio_with_openai(
                    chunk_file,
                    output_dirs["temp_transcriptions"],
                    openai_api_key,
                )
            else:
                transcription = transcribe_audio(
                    chunk_file,
                    output_dirs["temp_transcriptions"],
                    openai_api_key,
                )

            if transcription:
                transcriptions.append(transcription)
        transcription_time = time.time() - step_start_time
        logging.info(f"Transcription time: {format_seconds(transcription_time)}")

        # Log the contents of the temp_transcriptions directory
        list_directory_contents(output_dirs["temp_transcriptions"])

        # Verify that transcriptions match chunk files
        if len(transcriptions) != len(chunk_files):
            logging.warning(
                f"Mismatch between the number of transcriptions ({len(transcriptions)}) and chunk files ({len(chunk_files)})."
            )

        # Merge Transcriptions
        progress.update(file_task, description="Merging transcriptions", completed=80)
        logging.info("Merging transcriptions...")
        step_start_time = time.time()
        raw_merged_transcript = merge_raw_transcriptions(
            output_dirs["temp_transcriptions"],
            wav_file,
            output_dirs["transcriptions"],
        )
        merged_output = merge_transcriptions(
            raw_merged_transcript,
            diarization,
            wav_file,
            output_dirs["merged_output"],
            output_dirs["temp_chunks"],
        )
        merge_time = time.time() - step_start_time
        logging.info(f"Merge time: {format_seconds(merge_time)}")

        # Generate and save a waveform plot
        progress.update(file_task, description="Generating waveform plot", completed=90)
        try:
            logging.info("Generating waveform plot...")
            step_start_time = time.time()
            figure_filename = (
                os.path.basename(wav_file).rsplit(".", 1)[0] + "_waveform.png"
            )
            plot_waveform(wav_file, output_dirs["figures"], figure_filename)
            figure_time = time.time() - step_start_time
            logging.info(f"Figure generation time: {format_seconds(figure_time)}")
        except Exception as e:
            logging.error(f"Error generating waveform plot: {str(e)}")

        logging.info(f"Finished processing {file_path}")
        progress.update(file_task, completed=100)
        return merged_output

    except Exception as e:
        logging.error(f"Error processing audio file {file_path}: {str(e)}")
        raise

    finally:
        # Clean up temporary directories
        for temp_dir in [
            output_dirs["temp_wav"],
            output_dirs["temp_chunks"],
            output_dirs["temp_transcriptions"],
        ]:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory: {temp_dir}")


@safe_file_operation
def process_transcript_file(file_path, output_dirs, progress, file_task):
    """
    Process a single transcript file.

    Args:
        file_path (str): Path to the transcript file.
        output_dirs (dict): Dictionary of output directories.
        progress (rich.progress.Progress): Progress bar object.
        file_task (int): Task ID for the file being processed.

    Returns:
        str: Content of the transcript file.
    """
    logging.info(f"Processing transcript file: {file_path}")
    step_start_time = time.time()

    try:
        progress.update(file_task, description="Processing transcript", completed=50)
        transcript = read_file(file_path)
        output_file = os.path.join(
            output_dirs["merged_output"], os.path.basename(file_path)
        )
        write_file(output_file, transcript)

        process_time = time.time() - step_start_time
        logging.info(f"Transcript processing time: {format_seconds(process_time)}")
        logging.info(f"Copied transcript file to {output_file}")
        progress.update(file_task, completed=100)
        return transcript

    except Exception as e:
        logging.error(f"Error processing transcript file {file_path}: {str(e)}")
        raise


def process_files(
    input_dir,
    processed_dir,
    base_output_dir,
    hf_auth_token,
    openai_api_key,
    anthropic_api_key,
    summary_prompt_file,
    summarization_model,
    audio_files,
    transcript_files,
    openai_model,
    anthropic_model,
    openai_token_limit,
    anthropic_token_limit,
    use_openai,
    recording_date,
    recording_name,
    progress,
    file_task,
):
    """
    Processes audio files and transcript files.

    Args:
        input_dir (str): Directory containing input files.
        processed_dir (str): Directory to move processed files.
        base_output_dir (str): Base directory for output files.
        hf_auth_token (str): HuggingFace authentication token.
        openai_api_key (str): OpenAI API key.
        anthropic_api_key (str): Anthropic API key.
        summary_prompt_file (str): Path to the summary prompt file.
        summarization_model (str): Model to use for summarization.
        audio_files (list): List of audio files to process.
        transcript_files (list): List of transcript files to process.
        openai_model (str): OpenAI model name.
        anthropic_model (str): Anthropic model name.
        openai_token_limit (int): Token limit for OpenAI model.
        anthropic_token_limit (int): Token limit for Anthropic model.
        use_openai (bool): Whether to use OpenAI for transcription.
        recording_date (str): Custom recording date.
        recording_name (str): Custom recording name.
        progress (rich.progress.Progress): Progress bar object.
        file_task (int): Task ID for the file being processed.

    Returns:
        float: The total processing time for all files in seconds.
    """
    ensure_dir(input_dir)
    ensure_dir(processed_dir)
    ensure_dir(base_output_dir)

    total_size = 0
    total_length = 0
    start_time = time.time()

    logging.info("Starting to process input files...")

    for file_name in audio_files + transcript_files:
        file_path = os.path.join(input_dir, file_name)
        if recording_date is None:
            if is_audio_file(file_path):
                default_date = get_audio_metadata(file_path)
            else:
                default_date = time.strftime(
                    "%Y-%m-%d", time.gmtime(os.path.getmtime(file_path))
                )
        else:
            default_date = recording_date

        if recording_name is None:
            default_name = sanitize_filename(
                os.path.splitext(os.path.basename(file_path))[0]
            )
        else:
            default_name = sanitize_filename(recording_name)

        sanitized_recording_name = sanitize_filename(default_name)
        output_dirs = create_output_dirs(
            base_output_dir, default_date, sanitized_recording_name
        )

        file_start_time = time.time()
        try:
            if is_audio_file(file_path):
                merged_output = process_audio_file(
                    file_path,
                    output_dirs,
                    hf_auth_token,
                    openai_api_key,
                    use_openai,
                    progress,
                    file_task,
                )
            else:
                merged_output = process_transcript_file(
                    file_path, output_dirs, progress, file_task
                )

            if merged_output is not None:
                # Summarization
                progress.update(file_task, description="Summarizing", advance=10)
                logging.info(f"Starting summarization for {file_path}...")
                step_start_time = time.time()

                current_model = summarization_model
                current_token_limit = (
                    openai_token_limit
                    if summarization_model == openai_model
                    else anthropic_token_limit
                )
                current_api_key = (
                    openai_api_key
                    if summarization_model == openai_model
                    else anthropic_api_key
                )

                # Chunk the merged output if it exceeds the token limit
                estimated_tokens = estimate_tokens(merged_output)
                if estimated_tokens > current_token_limit:
                    logging.info(
                        "Merged output exceeds token limit. Chunking the text."
                    )

                    chunks = chunk_text(merged_output, current_token_limit)
                    summaries = []
                    for i, chunk in enumerate(chunks):
                        logging.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
                        chunk_summary = summarize_transcript(
                            chunk,
                            summary_prompt_file,
                            current_api_key,
                            current_model,
                            current_token_limit,
                        )
                        summaries.append(chunk_summary)
                        progress.update(file_task, advance=5 / len(chunks))

                    # Combine chunk summaries
                    combined_summary = "\n\n".join(summaries)
                    summary = summarize_transcript(
                        combined_summary,
                        summary_prompt_file,
                        current_api_key,
                        current_model,
                        current_token_limit,
                        is_final_summary=True,
                    )
                else:
                    summary = summarize_transcript(
                        merged_output,
                        summary_prompt_file,
                        current_api_key,
                        current_model,
                        current_token_limit,
                    )
                    progress.update(file_task, advance=10)

                summary_file = os.path.join(
                    output_dirs["summary"],
                    sanitize_filename(os.path.basename(file_path).rsplit(".", 1)[0])
                    + "_summary.md",
                )
                write_file(summary_file, summary)
                summarize_time = time.time() - step_start_time
                logging.info(f"Summarization time: {format_seconds(summarize_time)}")
                logging.info(f"Summarization completed and saved to {summary_file}")

            total_size += os.path.getsize(file_path)
            if is_audio_file(file_path):
                total_length += get_audio_length(file_path)

            # Move the processed file
            processed_file_path = os.path.join(
                processed_dir, os.path.basename(file_path)
            )
            shutil.move(file_path, processed_file_path)
            logging.info(f"Moved processed file to {processed_file_path}")

            file_time = time.time() - file_start_time
            logging.info(
                f"Total time taken for file {file_path}: {format_seconds(file_time)}"
            )

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            raise  # Re-raise the exception to be caught by the caller

    logging.info("Processing completed.")

    end_time = time.time()
    total_time = end_time - start_time

    logging.info(f"Total time taken: {format_seconds(total_time)}")
    logging.info(f"Total size of processed files: {total_size / (1024 * 1024):.2f} MB")
    logging.info(f"Total length of recordings: {format_seconds(total_length)}")
    logging.info(
        f"Number of files processed: {len(audio_files) + len(transcript_files)}"
    )

    return total_time  # Return the total processing time


if __name__ == "__main__":
    # This section is kept for backwards compatibility and testing
    import os
    from dotenv import load_dotenv
    from rich.progress import Progress

    load_dotenv()

    input_dir = "./audio/input"
    processed_dir = "./audio/processed"
    base_output_dir = "./output"
    hf_auth_token = os.getenv("HF_AUTH_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    summary_prompt_file = "./prompts/prompt.txt"
    summarization_model = os.getenv("OPENAI_MODEL")
    openai_model = os.getenv("OPENAI_MODEL")
    anthropic_model = os.getenv("ANTHROPIC_MODEL")
    openai_token_limit = int(os.getenv("OPENAI_MODEL_TOKEN_LIMIT", 8000))
    anthropic_token_limit = int(os.getenv("ANTHROPIC_MODEL_TOKEN_LIMIT", 100000))

    audio_files = [
        f for f in os.listdir(input_dir) if is_audio_file(os.path.join(input_dir, f))
    ]
    transcript_files = [
        f
        for f in os.listdir(input_dir)
        if is_transcript_file(os.path.join(input_dir, f))
    ]

    with Progress() as progress:
        file_task = progress.add_task("Processing files", total=100)
        process_files(
            input_dir,
            processed_dir,
            base_output_dir,
            hf_auth_token,
            openai_api_key,
            anthropic_api_key,
            summary_prompt_file,
            summarization_model,
            audio_files,
            transcript_files,
            openai_model,
            anthropic_model,
            openai_token_limit,
            anthropic_token_limit,
            use_openai=True,
            recording_date=None,
            recording_name=None,
            progress=progress,
            file_task=file_task,
        )
