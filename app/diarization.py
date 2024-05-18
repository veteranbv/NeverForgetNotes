from pyannote.audio import Pipeline
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def diarize_audio(audio_path, output_dir, hf_auth_token):
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_auth_token)
        diarization = pipeline(audio_path)
        
        # Save the diarization output to a file
        output_file = os.path.join(output_dir, os.path.basename(audio_path).replace('.wav', '_diarization.rttm'))
        with open(output_file, 'w') as f:
            diarization.write_rttm(f)
        
        logging.info(f"Diarization saved for {audio_path}")
        return diarization
    except Exception as e:
        logging.error(f"Error in diarization for {audio_path}: {e}")
        return None