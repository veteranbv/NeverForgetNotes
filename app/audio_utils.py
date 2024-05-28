import subprocess
import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from app.utils import ensure_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_wav(input_path, output_path):
    """
    Converts an audio file to WAV format using ffmpeg.
    
    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the converted WAV file.
    
    Raises:
        subprocess.CalledProcessError: If the conversion fails.
    """
    try:
        subprocess.run(
            ['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', output_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(f"Converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting file {input_path} to WAV: {e.stderr.decode().strip()}")
        raise

def get_audio_length(file_path):
    """
    Gets the length of an audio file in seconds.
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        float: The duration of the audio file in seconds.
    """
    try:
        with wave.open(file_path, 'rb') as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            duration = frames / float(rate)
            return duration
    except wave.Error as e:
        logging.error(f"Error getting audio length for {file_path}: {str(e)}")
        raise

def save_figure(fig, output_path, filename):
    """
    Saves a matplotlib figure to the specified output path.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        output_path (str): The directory to save the figure.
        filename (str): The filename for the saved figure.
    """
    ensure_dir(output_path)
    file_path = os.path.join(output_path, filename)
    fig.savefig(file_path)
    logging.info(f"Figure saved to {file_path}")

def plot_waveform(wav_file, output_path, filename):
    """
    Plots the waveform of a WAV file and saves it as an image.
    
    Args:
        wav_file (str): Path to the WAV file.
        output_path (str): The directory to save the waveform image.
        filename (str): The filename for the saved waveform image.
    """
    try:
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
            plt.close(fig)
    except wave.Error as e:
        logging.error(f"Error plotting waveform for {wav_file}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error plotting waveform for {wav_file}: {str(e)}")
        raise
