import os
import logging
from glob import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_transcriptions(transcriptions_path, diarization, audio_path, output_dir):
    """Merges transcriptions from speaker-specific chunks based on diarization.

    Args:
        transcriptions_path (str): Path to the merged raw transcripts
        diarization (pyannote.core.Annotation): Diarization annotation.
        audio_path (str): Path to the original audio file.
        output_dir (str): Directory to save the merged output.

    Returns:
        str: The merged transcribed text with speaker labels and timestamps.
    """

    merged_output = []
    chunk_files = sorted(glob(f'{output_dir}/*.wav'), key=lambda x: int(x.split('_chunk_')[1].split('_')[0]))
    with open(transcriptions_path) as f:
        transcriptions = f.readlines()
    i = 0
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = segment.start
        end = segment.end
        # Match transcription to chunk based on sorted order
        speaker_text = transcriptions[i]
        merged_output.append(f"[{start:.1f}s - {end:.1f}s] Speaker {speaker}: {speaker_text.strip()}")
        i += 1

    merged_output = "\n".join(merged_output)

    # Save the merged output to a file
    output_file = os.path.join(output_dir, os.path.basename(audio_path).replace('.wav', '_merged_output.txt'))
    with open(output_file, 'w') as f:
        f.write(merged_output)
        logging.info(f"Merged output saved to {output_file}")

    return merged_output

def merge_raw_transcriptions(temp_transcriptions_dir, audio_path, transcriptions_dir):
    """Merges raw transcriptions from temp directory into a single file.

    Args:
        temp_transcriptions_dir (str): Path to the temporary transcriptions directory.
        audio_path (str): Path to the original audio file.
        transcriptions_dir (str): Directory to save the merged transcription.

    Returns:
        str: The path to the merged raw transcript file. 
    """
    merged_text = []
    transcription_files = sorted(glob(f'{temp_transcriptions_dir}/*.txt'), 
                                 key=lambda x: int(x.split('_chunk_')[1].split('_')[0]) if '_chunk_' in x else 0) 
    for file in transcription_files:
        with open(file, 'r') as f:
            merged_text.append(f.read())

    merged_output_path = os.path.join(transcriptions_dir, os.path.basename(audio_path).replace('.wav', '_transcription.txt'))
    with open(merged_output_path, 'w') as f:
        f.write("\n".join(merged_text))
    logging.info(f"Raw transcriptions merged and saved to {merged_output_path}")

    return merged_output_path