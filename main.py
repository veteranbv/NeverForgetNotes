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

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

start_time = time.time()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory {directory}")

def convert_to_wav(input_path, output_path):
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', output_path], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
        # logging.info(f"Converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting file {input_path} to WAV: {e}")
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

def main():
    input_dir = './audio/input'
    output_dir = './output'
    transcriptions_dir = './output/transcriptions'
    temp_transcriptions_dir = './output/transcriptions/temp'
    diarizations_dir = './output/diarizations'
    merged_output_dir = './output/merged_output'
    processed_dir = './audio/processed'
    temp_dir = './audio/temp'
    summary_output_dir = './output/summary'
    figures_output_dir = './output/figures'
    hf_auth_token = os.getenv('HF_AUTH_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    summary_prompt_file = './prompts/summary_prompt.txt'

    ensure_dir(input_dir)
    ensure_dir(output_dir)
    ensure_dir(transcriptions_dir)
    ensure_dir(temp_transcriptions_dir)
    ensure_dir(diarizations_dir)
    ensure_dir(merged_output_dir)
    ensure_dir(processed_dir)
    ensure_dir(temp_dir)
    ensure_dir(summary_output_dir)
    ensure_dir(figures_output_dir)

    audio_files = glob(f'{input_dir}/*.m4a')
    if not audio_files:
        logging.info("No audio files found in the input directory.")
        return

    total_size = 0
    total_length = 0

    logging.info("Starting to process audio files...")
    for audio_file in audio_files:
        logging.info(f'Processing file: {audio_file}')
        audio_file_no_spaces = audio_file.replace(' ', '_')
        os.rename(audio_file, audio_file_no_spaces)
        wav_file = os.path.join(temp_dir, os.path.basename(audio_file_no_spaces).replace('.m4a', '.wav'))
        convert_to_wav(audio_file_no_spaces, wav_file)

        diarization = diarize_audio(wav_file, diarizations_dir, hf_auth_token)

        if diarization is not None:
            split_audio_by_diarization(wav_file, diarization, temp_dir)

            chunk_files = glob(f'{temp_dir}/*.wav')
            for chunk_file in tqdm(chunk_files, desc="Transcribing chunks", leave=False):
                transcription = transcribe_audio(chunk_file, temp_transcriptions_dir, openai_api_key)

            raw_merged_transcript = merge_raw_transcriptions(temp_transcriptions_dir, wav_file, transcriptions_dir)
            merged_output = merge_transcriptions(raw_merged_transcript, diarization, wav_file, merged_output_dir)

            # Summarize the merged transcript
            logging.info('Starting summarization...')
            summary = summarize_transcript(merged_output, summary_prompt_file, openai_api_key)
            summary_file = os.path.join(summary_output_dir, os.path.basename(wav_file).replace('.wav', '_summary.txt'))
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            logging.info(f"Summarization completed and saved to {summary_file}")

            # Generate and save a waveform plot
            figure_filename = os.path.basename(wav_file).replace('.wav', '_waveform.png')
            plot_waveform(wav_file, figures_output_dir, figure_filename)

        else:
            logging.warning(f"Skipping transcription and merge for {audio_file_no_spaces} due to missing diarization.")

        total_size += os.path.getsize(audio_file_no_spaces)
        total_length += get_audio_length(wav_file)
        os.rename(audio_file, os.path.join(processed_dir, os.path.basename(audio_file_no_spaces)))

    # Clean up temporary files 
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    for file in os.listdir(temp_transcriptions_dir):
        os.remove(os.path.join(temp_transcriptions_dir, file))
    logging.info("Processing completed.")
    
    end_time = time.time()
    total_time = end_time - start_time

    logging.info(f"Total time taken: {format_seconds(total_time)}")
    logging.info(f"Total size of processed files: {total_size / (1024 * 1024):.2f} MB")
    logging.info(f"Total length of recordings: {format_seconds(total_length)}")
    logging.info(f"Number of files processed: {len(audio_files)}")

if __name__ == "__main__":
    main()