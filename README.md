# Audio to Diarized Transcription Pipeline

![Banner](./img/top.jpeg)

This project provides an automated pipeline for diarizing and transcribing audio recordings, with a focus on Apple Voice Memos (.m4a files). The pipeline utilizes Whisper or OpenAI's API for audio transcription, pyannote.audio for speaker diarization, and GPT-4 or Claude for generating summaries of the merged transcripts. The output includes a merged transcript with speaker labels, timestamps, and a summary.

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
  - [Audio Utils](#audio-utils)
  - [Utilities](#utilities)
- [Configuration](#configuration)
- [Testing](#testing)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)
- [Meeting Summary Prompt Example](#meeting-summary-prompt-example)

## Installation

1. Clone the repository:

   ```zsh
   git clone https://github.com/veteranbv/NeverForgetNotes.git
   ```

2. Change to the project directory:

   ```zsh
   cd NeverForgetNotes
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
     ANTHROPIC_API_KEY=your_anthropic_api_key
     OPENAI_MODEL=gpt-4-turbo
     OPENAI_MODEL_TOKEN_LIMIT=8000
     ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
     ANTHROPIC_MODEL_TOKEN_LIMIT=100000
     ```

   - Replace `your_openai_api_key`, `your_hugging_face_auth_token`, and `your_anthropic_api_key` with your actual API keys.

## Usage

1. Place your audio files (in `.m4a` format) in the `audio/input` directory.

2. Run the main script:

   ```zsh
   python main.py
   ```

3. The script will prompt you to choose global settings or individual settings for each file. You'll be asked about:
   - Transcription method (OpenAI API or local Whisper)
   - Summarization model (GPT-4 or Claude)
   - Prompt selection for summarization
   - Custom recording details (optional)

4. The script will process the audio files, perform speaker diarization, transcribe them, generate merged transcripts with speaker labels and timestamps, and create a summary.

5. The output files will be saved in the `output` directory, organized by recording date and name:
   - `output/<recording_date>-<recording_name>/transcriptions`: Contains the raw transcriptions for each audio file.
   - `output/<recording_date>-<recording_name>/diarizations`: Contains the diarization output for each audio file.
   - `output/<recording_date>-<recording_name>/merged_output`: Contains the merged transcripts with speaker labels and timestamps.
   - `output/<recording_date>-<recording_name>/summary`: Contains the summaries of the merged transcripts.
   - `output/<recording_date>-<recording_name>/figures`: Contains the waveform figures for each audio file.

6. The processed audio files will be moved to the `audio/processed` directory.

## Project Structure

```
NeverForgetNotes/
├── LICENSE
├── README.md
├── app/
│   ├── audio_processing.py
│   ├── audio_utils.py
│   ├── diarization.py
│   ├── merge.py
│   ├── split.py
│   ├── summarize.py
│   ├── transcription.py
│   └── utils.py
├── audio/
│   ├── input/
│   ├── not_processed/
│   └── processed/
├── img/
│   └── top.jpeg
├── main.py
├── output/
├── prompts/
│   ├── library/
│   └── prompt.txt
├── requirements.txt
├── temp/
│   ├── chunks/
│   ├── transcriptions/
│   └── wav_files/
└── test/
    ├── create_testdata.py
    └── data/
```

## Modules

### Transcription

The `transcription.py` module handles audio transcription using either Whisper or OpenAI's API. It provides functions for transcribing audio chunks and saving the transcriptions to files.

### Diarization

The `diarization.py` module performs speaker diarization using pyannote.audio. It provides the `diarize_audio` function that takes an audio file, performs diarization, and saves the diarization output to a file.

### Merging

The `merge.py` module handles the merging of transcriptions and diarization outputs. It provides two main functions:

- `merge_transcriptions`: Merges the transcriptions from speaker-specific chunks based on the diarization output.
- `merge_raw_transcriptions`: Merges the raw transcriptions from the temporary directory into a single file.

### Splitting

The `split.py` module handles the splitting of audio files into speaker-specific chunks based on the diarization output. It provides the `split_audio_by_diarization` function that takes an audio file and diarization output, splits the audio into chunks, and saves the chunks to files.

### Summarizing

The `summarize.py` module handles the generation of summaries for the merged transcripts using either GPT-4 or Claude. It provides the `summarize_transcript` function that takes the merged transcript, a prompt file, and the appropriate API key, and returns the generated summary.

### Audio Utils

The `audio_utils.py` module provides utility functions for audio processing, including converting audio formats, getting audio length, and plotting waveforms.

### Utilities

The `utils.py` module provides various utility functions for directory creation, file handling, formatting, and other common operations used throughout the project.

## Configuration

The project uses environment variables for configuration. The following variables need to be set in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key for using GPT-4.
- `HF_AUTH_TOKEN`: Your Hugging Face authentication token for using pyannote.audio.
- `ANTHROPIC_API_KEY`: Your Anthropic API key for using Claude.
- `OPENAI_MODEL`: The OpenAI model to use for summarization (e.g., "gpt-4-turbo").
- `OPENAI_MODEL_TOKEN_LIMIT`: The token limit for the OpenAI model.
- `ANTHROPIC_MODEL`: The Anthropic model to use for summarization (e.g., "claude-3-5-sonnet-20240620").
- `ANTHROPIC_MODEL_TOKEN_LIMIT`: The token limit for the Anthropic model.

## Testing

The `test/` directory contains test data and scripts for validating the functionality of the pipeline. You can use the `create_testdata.py` script to generate test audio files from your input files.

To run tests, you can execute individual module scripts (e.g., `python -m app.transcription`) or create dedicated test scripts for more comprehensive testing.

## Logging

The project uses Python's built-in `logging` module for logging information and errors. Log messages are formatted with timestamps and saved to rotating log files in the `logs/` directory. You can adjust the logging level and configuration in the `setup_logging` function in `main.py`.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Meeting Summary Prompt Example

To generate a meeting summary, you can use the following prompt example and put it into a prompt.txt file in the prompts directory. The pipeline will use this prompt to generate a summary of the meeting transcript:

```markdown
Here is the transcript of the meeting:

<transcript>
{{TRANSCRIPT}}
</transcript>

Please read through the transcript carefully and complete the following steps:

1. Identify and extract the key points, decisions, action items (with owners if available), and any other critical information discussed during the meeting. 

2. Organize and categorize this extracted information into relevant sections such as main topics, decisions, action items, etc.

3. In the <scratchpad> section below, write out your thought process and the information you extracted, organized into appropriate categories:

<scratchpad>
(Write your thought process and extracted information here, organized into relevant categories)
</scratchpad>

4. Now, using the information in your scratchpad, please compose a concise yet comprehensive summary of the meeting that captures the essence and key outcomes. Structure and format the summary as follows:

---

**Meeting Summary for [Meeting Name] ([Date])**

**Quick Recap**

[Summary of main topics covered]

**Next Steps**

- [Owner] will [action item]
- [Owner] will [action item]
- [Owner] will [action item]

**Summary**

**[Topic 1]**

[Detailed discussion, decisions, and action items]

**[Topic 2]**

[Detailed discussion, decisions, and action items]

**[Topic 3]**

[Detailed discussion, decisions, and action items]

**[Additional Topics]**

[Any other critical information]

---

5. Please output your final summary inside <summary> tags.

<summary>
(Output your final summary here)
</summary>
```
