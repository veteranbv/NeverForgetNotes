import os
import logging
from glob import glob

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def merge_transcriptions(
    transcriptions_path, diarization, audio_path, output_dir, temp_chunks_dir
):
    """
    Merges transcriptions from speaker-specific chunks based on diarization.

    Args:
        transcriptions_path (str): Path to the merged raw transcripts.
        diarization (Annotation): Diarization annotation.
        audio_path (str): Path to the original audio file.
        output_dir (str): Directory to save the merged output.
        temp_chunks_dir (str): Directory containing the audio chunks.

    Returns:
        str: The merged transcribed text with speaker labels and timestamps.
    """
    try:
        merged_output = []
        chunk_files = sorted(
            glob(os.path.join(temp_chunks_dir, "*.wav")),
            key=lambda x: int(x.split("_chunk_")[1].split("_")[0]),
        )
        logging.info(f"Found chunk files: {chunk_files}")

        with open(transcriptions_path) as f:
            transcriptions = f.readlines()

        if len(transcriptions) != len(chunk_files):
            logging.warning(
                f"Mismatch between the number of transcriptions ({len(transcriptions)}) and chunk files ({len(chunk_files)})."
            )

        transcription_idx = 0
        for i, (segment, _, speaker) in enumerate(
            diarization.itertracks(yield_label=True)
        ):
            start = segment.start
            end = segment.end
            if transcription_idx < len(transcriptions):
                speaker_text = transcriptions[transcription_idx].strip()
                transcription_idx += 1
            else:
                speaker_text = "[Transcription missing]"

            merged_output.append(
                f"[{start:.1f}s - {end:.1f}s] {speaker}: {speaker_text}"
            )

        merged_output = "\n".join(merged_output)

        output_file = os.path.join(
            output_dir,
            os.path.basename(audio_path).replace(".wav", "_merged_output.txt"),
        )
        with open(output_file, "w") as f:
            f.write(merged_output)
            logging.info(f"Merged output saved to {output_file}")

        return merged_output
    except Exception as e:
        logging.error(f"Error merging transcriptions: {str(e)}")
        raise


def merge_raw_transcriptions(temp_transcriptions_dir, audio_path, transcriptions_dir):
    """
    Merges raw transcription text files into a single transcription file.

    Args:
        temp_transcriptions_dir (str): Directory containing temporary transcription files.
        audio_path (str): Path to the original audio file.
        transcriptions_dir (str): Directory to save the merged transcription.

    Returns:
        str: Path to the merged transcription file.
    """
    try:
        merged_text = []
        transcription_files = sorted(
            glob(os.path.join(temp_transcriptions_dir, "*.txt")),
            key=lambda x: int(x.split("_chunk_")[1].split("_")[0])
            if "_chunk_" in x
            else 0,
        )
        for file in transcription_files:
            with open(file, "r") as f:
                merged_text.append(f.read())

        merged_output_path = os.path.join(
            transcriptions_dir,
            os.path.basename(audio_path).replace(".wav", "_transcription.txt"),
        )
        with open(merged_output_path, "w") as f:
            f.write("\n".join(merged_text))
        logging.info(f"Raw transcriptions merged and saved to {merged_output_path}")

        return merged_output_path
    except Exception as e:
        logging.error(f"Error merging raw transcriptions: {str(e)}")
        raise
