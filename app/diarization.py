from pyannote.audio import Pipeline
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def diarize_audio(audio_path, output_dir, hf_auth_token):
    """
    Performs speaker diarization on an audio file using Pyannote.
    
    Args:
        audio_path (str): Path to the input audio file.
        output_dir (str): Directory to save the diarization output.
        hf_auth_token (str): Hugging Face authentication token.
    
    Returns:
        pyannote.core.Annotation: The diarization annotation.
    """
    try:
        # Initialize the diarization pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_auth_token)
        # Perform diarization
        diarization = pipeline(audio_path)
        
        # Save the diarization output to a file
        output_file = os.path.join(output_dir, os.path.basename(audio_path).replace('.wav', '_diarization.rttm'))
        with open(output_file, 'w') as f:
            diarization.write_rttm(f)
        
        logging.info(f"Diarization saved for {audio_path} to {output_file}")
        return diarization
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {audio_path} - {str(fnf_error)}")
        raise
    except Exception as e:
        logging.error(f"Error in diarization for {audio_path}: {str(e)}")
        raise