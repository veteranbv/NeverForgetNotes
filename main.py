import os
from glob import glob
from tqdm import tqdm
import logging
import subprocess
from app.transcription import transcribe_audio
from app.diarization import diarize_audio
from app.merge import merge_transcriptions, merge_raw_transcriptions
from app.split import split_audio_by_diarization
from app.summarize import summarize_transcript

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory {directory}")

def convert_to_wav(input_path, output_path):
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', output_path], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
        logging.info(f"Converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting file {input_path} to WAV: {e}")
        raise

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

    audio_files = glob(f'{input_dir}/*.m4a')
    if not audio_files:
        logging.info("No audio files found in the input directory.")
        return

    for audio_file in tqdm(audio_files, desc="Processing audio files"):
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
            summary = summarize_transcript(merged_output, summary_prompt_file, openai_api_key)
            summary_file = os.path.join(summary_output_dir, os.path.basename(wav_file).replace('.wav', '_summary.txt'))
            with open(summary_file, 'w') as f:
                f.write(summary)
            logging.info(f"Summary saved to {summary_file}")

        else:
            logging.warning(f"Skipping transcription and merge for {audio_file_no_spaces} due to missing diarization.")

        os.rename(audio_file, os.path.join(processed_dir, os.path.basename(audio_file_no_spaces)))

    # Clean up temporary files 
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    for file in os.listdir(temp_transcriptions_dir):
        os.remove(os.path.join(temp_transcriptions_dir, file))
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()