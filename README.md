# Audio to Diarized Transcription Pipeline

![Banner](./img/top.jpeg)

This project provides an automated pipeline for diarizing and transcribing audio recordings, specifically focusing on Apple Voice Memos (or .m4a files). The pipeline utilizes Whisper for audio transcription, pyannote.audio for speaker diarization, and GPT-4o for generating summaries of the merged transcripts. The output includes a merged transcript with speaker labels, timestamps, and a summary.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [Transcription](#transcription)
  - [Diarization](#diarization)
  - [Merging](#merging)
  - [Splitting](#splitting)
  - [Summarizing](#summarizing)
- [Configuration](#configuration)
- [Testing](#testing)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```zsh
   git clone https://github.com/veteranbv/M4A_to_Diarized_Transcription.git
   ```

2. Change to the project directory:

   ```zsh
   cd M4A_to_Diarized_Transcription
   ```

3. Create a virtual environment:

   ```zsh
   python -m venv venv
   ```

4. Activate the virtual environment:

   - For Windows:

     ```zsh
     venv\Scripts\activate
     ```

   - For Unix or MacOS:

     ```zsh
     source venv/bin/activate
     ```

5. Install the required dependencies:

   ```zsh
   pip install -r requirements.txt
   ```

6. Set up the necessary environment variables:
   - Create a `.env` file in the project root.
   - Add the following variables to the `.env` file:

     ```txt
     OPENAI_API_KEY=your_openai_api_key
     HF_AUTH_TOKEN=your_hugging_face_auth_token
     ```

   - Replace `your_openai_api_key` and `your_hugging_face_auth_token` with your actual API keys.

## Usage

1. Place your audio files (in `.m4a` format) in the `audio/input` directory.

2. Run the main script:

   ```zsh
   python main.py
   ```

3. The script will process the audio files, perform speaker diarization, transcribe them, generate merged transcripts with speaker labels and timestamps, and create a summary.

4. The output files will be saved in the `output` directory, organized by recording date and name:
   - `output/<recording_date>-<recording_name>/transcriptions`: Contains the raw transcriptions for each audio file.
   - `output/<recording_date>-<recording_name>/diarizations`: Contains the diarization output for each audio file.
   - `output/<recording_date>-<recording_name>/merged_output`: Contains the merged transcripts with speaker labels and timestamps.
   - `output/<recording_date>-<recording_name>/summary`: Contains the summaries of the merged transcripts.
   - `output/<recording_date>-<recording_name>/figures`: Contains the waveform figures for each audio file.

5. The processed audio files will be moved to the `audio/processed` directory.

## Project Structure

   ```zsh
   ./M4A_to_Diarized_Transcription
   ├── LICENSE
   ├── README.md
   ├── app
   │   ├── diarization.py
   │   ├── merge.py
   │   ├── split.py
   │   ├── summarize.py
   │   └── transcription.py
   ├── audio
   │   ├── input
   │   └── processed
   ├── img
   │   └── top.jpeg
   ├── main.py
   ├── output
   ├── prompts
   │   └── summary_prompt.txt
   ├── requirements.txt
   ├── temp
   │   ├── chunks
   │   ├── transcriptions
   │   └── wav_files
   └── test
      └── data
   ```

- `requirements.txt`: Lists the required dependencies for the project.
- `README.md`: Provides an overview and instructions for the project.
- `main.py`: The main script that orchestrates the workflow.
- `app/`: Contains the modules for transcription, diarization, merging, splitting, and summarizing.
- `prompts/`: Contains the prompt file for the GPT-4o model.
- `test/`: Contains test data and scripts.
- `output/`: Stores the output files generated by the pipeline.
- `audio/`: Stores the input and processed audio files.
- `temp/`: Stores temporary files during processing.

## Modules

### Transcription

The `transcription.py` module handles the audio transcription using Whisper. It provides the `transcribe_audio` function that takes an audio chunk, transcribes it, and saves the transcription to a file.

### Diarization

The `diarization.py` module performs speaker diarization using pyannote.audio. It provides the `diarize_audio` function that takes an audio file, performs diarization, and saves the diarization output to a file.

### Merging

The `merge.py` module handles the merging of transcriptions and diarization outputs. It provides two functions:

- `merge_transcriptions`: Merges the transcriptions from speaker-specific chunks based on the diarization output.
- `merge_raw_transcriptions`: Merges the raw transcriptions from the temporary directory into a single file.

### Splitting

The `split.py` module handles the splitting of audio files into speaker-specific chunks based on the diarization output. It provides the `split_audio_by_diarization` function that takes an audio file and diarization output, splits the audio into chunks, and saves the chunks to files.

### Summarizing

The `summarize.py` module handles the generation of summaries for the merged transcripts using the GPT-4o model. It provides the `summarize_transcript` function that takes the merged transcript, a prompt file, and the OpenAI API key, and returns the generated summary.

## Configuration

The project uses environment variables for configuration. The following variables need to be set in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key for using GPT-4o.
- `HF_AUTH_TOKEN`: Your Hugging Face authentication token for using pyannote.audio.

## Testing

The `test/` directory contains test data and scripts for validating the functionality of the pipeline. You can add your own test cases and run them to ensure the correctness of the modules.

## Logging

The project uses Python's built-in `logging` module for logging information and errors. The log messages are formatted with timestamps and logged to the console. You can adjust the logging level and add file logging if needed.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
