import os
from glob import glob
from tqdm import tqdm
import logging
import subprocess
import time
from app.transcription import transcribe_audio
from app.diarization import diarize_audio
from app.merge import merge_transcriptions, merge_raw_transcriptions
from app.split import split_audio_by_diarization
from app.summarize import summarize_transcript
import wave
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

start_time = time.time()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        # logging.info(f"Created directory {directory}")

def convert_to_wav(input_path, output_path):
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', output_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(f"Converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting file {input_path} to WAV: {e.stderr.decode().strip()}")
        raise

def get_audio_length(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)
        return duration

def format_seconds(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def save_figure(fig, output_path, filename):
    ensure_dir(output_path)
    file_path = os.path.join(output_path, filename)
    fig.savefig(file_path)
    logging.info(f"Figure saved to {file_path}")

def plot_waveform(wav_file, output_path, filename):
    with wave.open(wav_file, 'rb') as audio_file:
        signal = audio_file.readframes(-1)
        signal = np.frombuffer(signal, dtype=np.int16)
        rate = audio_file.getframerate()
        time = np.linspace(0., len(signal) / rate, num=len(signal))

        fig, ax = plt.subplots()
        ax.plot(time, signal)
        ax.set_title(f'Waveform of {os.path.basename(wav_file)}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        save_figure(fig, output_path, filename)
        plt.close(fig)  # Close the figure to free memory

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name).lower()

def prompt_for_recording_details(default_date, default_name, filename):
    use_custom_details = input(f"Do you want to provide a custom recording date and meeting name for {filename}? (y/n): ").strip().lower()
    if use_custom_details == 'y':
        recording_date = input("Enter the recording date (YYYY-MM-DD): ")
        recording_name = input("Enter the name for the recording: ")
        try:
            normalized_date = datetime.strptime(recording_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            logging.error("Invalid date format. Please use YYYY-MM-DD.")
            raise
    else:
        normalized_date = default_date
        recording_name = default_name
    sanitized_name = sanitize_filename(recording_name)
    return normalized_date, sanitized_name

def create_output_dirs(base_output_dir, normalized_date, recording_name):
    output_dir = os.path.join(base_output_dir, f"{normalized_date}-{recording_name}")
    dirs = {
        'transcriptions': os.path.join(output_dir, 'transcriptions'),
        'diarizations': os.path.join(output_dir, 'diarizations'),
        'merged_output': os.path.join(output_dir, 'merged_output'),
        'summary': os.path.join(output_dir, 'summary'),
        'figures': os.path.join(output_dir, 'figures'),
        'temp_wav': './temp/wav_files',  # Centralized temp directory for wav files
        'temp_chunks': './temp/chunks',  # Centralized temp directory for chunks
        'temp_transcriptions': './temp/transcriptions'  # Centralized temp directory for transcriptions
    }
    for directory in dirs.values():
        ensure_dir(directory)
    return dirs

def get_audio_metadata(file_path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format_tags=date', '-of', 'default=nw=1', file_path], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        date = result.stdout.strip().split('=')[-1]
        return datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
    except:
        return datetime.now().strftime('%Y-%m-%d')

def main():
    input_dir = './audio/input'
    processed_dir = './audio/processed'
    base_output_dir = './output'
    hf_auth_token = os.getenv('HF_AUTH_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    summary_prompt_file = './prompts/summary_prompt.txt'

    ensure_dir(input_dir)
    ensure_dir(processed_dir)
    ensure_dir(base_output_dir)

    audio_files = glob(f'{input_dir}/*.m4a')
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

    logging.info("Starting to process audio files...")
    for idx, audio_file in enumerate(audio_files):
        file_start_time = time.time()
        recording_date, recording_name = recording_details[idx]
        output_dirs = create_output_dirs(base_output_dir, recording_date, recording_name)

        logging.info(f'Processing file: {audio_file}')
        audio_file_no_spaces = audio_file.replace(' ', '_')
        os.rename(audio_file, audio_file_no_spaces)
        wav_file = os.path.join(output_dirs['temp_wav'], os.path.basename(audio_file_no_spaces).replace('.m4a', '.wav'))

        step_start_time = time.time()
        convert_to_wav(audio_file_no_spaces, wav_file)
        convert_time = time.time() - step_start_time

        step_start_time = time.time()
        diarization = diarize_audio(wav_file, output_dirs['diarizations'], hf_auth_token)
        diarization_time = time.time() - step_start_time

        if diarization is not None:
            step_start_time = time.time()
            split_audio_by_diarization(wav_file, diarization, output_dirs['temp_chunks'])
            split_time = time.time() - step_start_time

            step_start_time = time.time()
            chunk_files = glob(f'{output_dirs["temp_chunks"]}/*.wav')
            for chunk_file in tqdm(chunk_files, desc="Transcribing chunks", leave=False):
                transcription = transcribe_audio(chunk_file, output_dirs['temp_transcriptions'], openai_api_key)
            transcription_time = time.time() - step_start_time

            step_start_time = time.time()
            raw_merged_transcript = merge_raw_transcriptions(output_dirs['temp_transcriptions'], wav_file, output_dirs['transcriptions'])
            merged_output = merge_transcriptions(raw_merged_transcript, diarization, wav_file, output_dirs['merged_output'])
            merge_time = time.time() - step_start_time

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

if __name__ == "__main__":
    main()