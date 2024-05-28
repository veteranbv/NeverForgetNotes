import os
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_prompt(prompt_file):
    """
    Loads the prompt template from a file.
    
    Args:
        prompt_file (str): Path to the prompt file.
    
    Returns:
        str: The prompt template.
    """
    try:
        with open(prompt_file, 'r') as file:
            prompt = file.read()
        return prompt
    except FileNotFoundError as e:
        logging.error(f"Prompt file not found: {prompt_file} - {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading prompt file {prompt_file}: {str(e)}")
        raise

def summarize_transcript(transcript, prompt_file, openai_api_key):
    """
    Summarizes a transcript using OpenAI's API.
    
    Args:
        transcript (str): The transcript to summarize.
        prompt_file (str): Path to the prompt template file.
        openai_api_key (str): OpenAI API key.
    
    Returns:
        str: The summary of the transcript.
    """
    try:
        client = OpenAI(api_key=openai_api_key)
        prompt = load_prompt(prompt_file).replace("{{TRANSCRIPT}}", transcript)

        logging.info("Sending request to OpenAI API for summarization...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        summary = response.choices[0].message.content.strip()
        logging.info("Summarization completed successfully.")
        return summary
    except Exception as e:
        logging.error(f"Error in summarization: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    example_transcript = "This is a test transcript."
    example_prompt_file = "prompts/prompt.txt"
    example_openai_api_key = os.getenv('OPENAI_API_KEY')

    try:
        summary = summarize_transcript(example_transcript, example_prompt_file, example_openai_api_key)
        print(summary)
    except Exception as e:
        logging.error(f"Failed to summarize transcript: {str(e)}")