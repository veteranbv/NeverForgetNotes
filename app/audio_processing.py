import logging
from glob import glob
from tqdm import tqdm
import os
import time

from app.utils import ensure_dir, format_seconds, prompt_for_recording_details, create_output_dirs, get_audio_metadata
from app.audio_utils import convert_to_wav, get_audio_length, plot_waveform
from app.transcription import transcribe_audio
from app.diarization import diarize_audio
from app.merge import merge_transcriptions, merge_raw_transcriptions
from app.split import split_audio_by_diarization
from app.summarize import summarize_transcript

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            logging.info(f"  - {os.path.join(root, name)}")

def process_audio_files(input_dir, processed_dir, base_output_dir, hf_auth_token, openai_api_key, summary_prompt_file):
    """
    Processes audio files by converting, diarizing, splitting, transcribing, merging, and summarizing them.
    
    Args:
        input_dir (str): Directory containing the input audio files.
        processed_dir (str): Directory to move processed audio files.
        base_output_dir (str): Base directory to store output files.
        hf_auth_token (str): Hugging Face API token for diarization.
        openai_api_key (str): OpenAI API key for transcription and summarization.
        summary_prompt_file (str): File containing the prompt for summarization.
    
    Returns:
        None
    """
    ensure_dir(input_dir)
    ensure_dir(processed_dir)
    ensure_dir(base_output_dir)

    audio_files = glob(os.path.join(input_dir, '*.m4a'))
    if not audio_files:
        logging.info("No audio files found in the input directory.")
        return

    recording_details = []
    for audio_file in audio_files:
        default_date = get_audio_metadata(audio_file)
        default_name = os.path.splitext(os.path.basename(audio_file))[0]
        recording_details.append(prompt_for_recording_details(default_date, default_name, audio_file))

    total_size = 0
    total_length = 0
    start_time = time.time()

    logging.info("Starting to process audio files...")
    for idx, audio_file in enumerate(audio_files):
        file_start_time = time.time()
        recording_date, recording_name = recording_details[idx]
        output_dirs = create_output_dirs(base_output_dir, recording_date, recording_name)

        logging.info(f'Processing file: {audio_file}')
        audio_file_no_spaces = audio_file.replace(' ', '_')
        os.rename(audio_file, audio_file_no_spaces)
        wav_file = os.path.join(output_dirs['temp_wav'], os.path.basename(audio_file_no_spaces).replace('.m4a', '.wav'))

        try:
            # Convert to WAV
            step_start_time = time.time()
            convert_to_wav(audio_file_no_spaces, wav_file)
            convert_time = time.time() - step_start_time

            # Diarization
            step_start_time = time.time()
            diarization = diarize_audio(wav_file, output_dirs['diarizations'], hf_auth_token)
            diarization_time = time.time() - step_start_time

            if diarization is not None:
                # Split Audio by Diarization
                step_start_time = time.time()
                chunk_files = split_audio_by_diarization(wav_file, diarization, output_dirs['temp_chunks'])
                split_time = time.time() - step_start_time

                if not chunk_files:
                    logging.error(f"No audio chunks found in {output_dirs['temp_chunks']}")
                    continue

                # Log the contents of the temp_chunks directory
                list_directory_contents(output_dirs['temp_chunks'])

                # Transcription
                step_start_time = time.time()
                transcriptions = []
                for chunk_file in tqdm(chunk_files, desc="Transcribing chunks", leave=False):
                    transcription = transcribe_audio(chunk_file, output_dirs['temp_transcriptions'], openai_api_key)
                    if transcription:
                        transcriptions.append(transcription)
                transcription_time = time.time() - step_start_time

                # Log the contents of the temp_transcriptions directory
                list_directory_contents(output_dirs['temp_transcriptions'])

                # Verify that transcriptions match chunk files
                if len(transcriptions) != len(chunk_files):
                    logging.warning(f"Mismatch between the number of transcriptions ({len(transcriptions)}) and chunk files ({len(chunk_files)}).")

                # Merge Transcriptions
                step_start_time = time.time()
                raw_merged_transcript = merge_raw_transcriptions(output_dirs['temp_transcriptions'], wav_file, output_dirs['transcriptions'])
                merged_output = merge_transcriptions(raw_merged_transcript, diarization, wav_file, output_dirs['merged_output'])
                merge_time = time.time() - step_start_time

                # Summarization
                step_start_time = time.time()
                logging.info('Starting summarization...')
                summary = summarize_transcript(merged_output, summary_prompt_file, openai_api_key)
                summary_file = os.path.join(output_dirs['summary'], os.path.basename(wav_file).replace('.wav', '_summary.md'))
                with open(summary_file, 'w') as f:
                    f.write(summary)
                summarize_time = time.time() - step_start_time

                logging.info(f"Summarization completed and saved to {summary_file}")

                # Generate and save a waveform plot
                step_start_time = time.time()
                figure_filename = os.path.basename(wav_file).replace('.wav', '_waveform.png')
                plot_waveform(wav_file, output_dirs['figures'], figure_filename)
                figure_time = time.time() - step_start_time

            else:
                logging.warning(f"Skipping transcription and merge for {audio_file_no_spaces} due to missing diarization.")

            total_size += os.path.getsize(audio_file_no_spaces)
            total_length += get_audio_length(wav_file)
            os.rename(audio_file_no_spaces, os.path.join(processed_dir, os.path.basename(audio_file_no_spaces)))

            file_time = time.time() - file_start_time

            logging.info(f"Time taken for file {audio_file}: {format_seconds(file_time)}")
            logging.info(f"  - Convert time: {format_seconds(convert_time)}")
            logging.info(f"  - Diarization time: {format_seconds(diarization_time)}")
            logging.info(f"  - Split time: {format_seconds(split_time)}")
            logging.info(f"  - Transcription time: {format_seconds(transcription_time)}")
            logging.info(f"  - Merge time: {format_seconds(merge_time)}")
            logging.info(f"  - Summarize time: {format_seconds(summarize_time)}")
            logging.info(f"  - Figure generation time: {format_seconds(figure_time)}")

        except Exception as e:
            logging.error(f"Error processing file {audio_file}: {str(e)}")
            continue

    # Clean up temporary files 
    temp_dir = './temp'
    for subdir in ['wav_files', 'chunks', 'transcriptions']:
        subdir_path = os.path.join(temp_dir, subdir)
        if os.path.exists(subdir_path):
            for file in os.listdir(subdir_path):
                os.remove(os.path.join(subdir_path, file))
    logging.info("Processing completed.")
    
    end_time = time.time()
    total_time = end_time - start_time

    logging.info(f"Total time taken: {format_seconds(total_time)}")
    logging.info(f"Total size of processed files: {total_size / (1024 * 1024):.2f} MB")
    logging.info(f"Total length of recordings: {format_seconds(total_length)}")
    logging.info(f"Number of files processed: {len(audio_files)}")

if __name__ == '__main__':
    input_dir = './audio/input'
    processed_dir = './audio/processed'
    base_output_dir = './output'
    hf_auth_token = os.getenv('HF_AUTH_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    summary_prompt_file = './prompts/prompt.txt'
    
    process_audio_files(input_dir, processed_dir, base_output_dir, hf_auth_token, openai_api_key, summary_prompt_file)
