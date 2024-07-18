import os
import logging
from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import tiktoken
from app.utils import chunk_text, read_file, safe_file_operation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@safe_file_operation
def load_prompt(prompt_file):
    """
    Loads the prompt template from a file.

    Args:
        prompt_file (str): Path to the prompt file.

    Returns:
        str: The prompt template.
    """
    return read_file(prompt_file)


def count_tokens(text, model):
    """
    Count the number of tokens in the given text.

    Args:
        text (str): The text to count tokens for.
        model (str): The model to use for token counting.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Error counting tokens: {str(e)}. Using fallback estimation.")
        return len(text.split()) * 1.3  # Fallback estimation


def summarize_with_openai(prompt, api_key, model):
    """
    Summarize text using OpenAI's API.

    Args:
        prompt (str): The prompt containing the text to summarize.
        api_key (str): OpenAI API key.
        model (str): The OpenAI model to use for summarization.

    Returns:
        str: The generated summary.
    """
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in OpenAI summarization: {str(e)}")
        raise


def summarize_with_anthropic(prompt, api_key, model, max_tokens=8192):
    """
    Summarize text using Anthropic's Claude API with extended token limit.

    Args:
        prompt (str): The prompt containing the text to summarize.
        api_key (str): Anthropic API key.
        model (str): The Anthropic model to use for summarization.
        max_tokens (int): Maximum number of tokens for the output. Defaults to 8192.

    Returns:
        str: The generated summary.
    """
    client = Anthropic(
        api_key=api_key,
        default_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": f"{HUMAN_PROMPT} {prompt}"},
                {"role": "assistant", "content": f"{AI_PROMPT}"},
            ],
        )
        return response.content[0].text
    except Exception as e:
        logging.error(f"Error in Anthropic summarization: {str(e)}")
        raise


def summarize_transcript(
    transcript, prompt_file, api_key, model, token_limit, is_final_summary=False
):
    """
    Summarizes a transcript using either OpenAI's API or Anthropic's Claude.

    Args:
        transcript (str): The transcript to summarize.
        prompt_file (str): Path to the prompt template file.
        api_key (str): API key for the chosen model.
        model (str): The model to use for summarization (e.g., "gpt-4" or "claude-2").
        token_limit (int): The token limit for the chosen model.
        is_final_summary (bool): Whether this is the final summary of chunked summaries.

    Returns:
        str: The summary of the transcript.
    """
    try:
        prompt_template = load_prompt(prompt_file)
        chunk_limit = token_limit - 1000  # Leave room for the prompt
        chunks = chunk_text(transcript, chunk_limit)
        summaries = []

        for i, chunk in enumerate(chunks):
            chunk_prompt = prompt_template.replace("{{TRANSCRIPT}}", chunk)
            if i > 0 or is_final_summary:
                chunk_prompt += "\n\nNote: This is a continuation or combination of previous summaries. Please provide a coherent and comprehensive summary."

            if "gpt" in model.lower():
                summary = summarize_with_openai(chunk_prompt, api_key, model)
            elif "claude" in model.lower():
                summary = summarize_with_anthropic(chunk_prompt, api_key, model)
            else:
                raise ValueError(f"Unsupported model: {model}")

            summaries.append(summary)

        # Combine summaries if there are multiple chunks
        if len(summaries) > 1:
            combined_summary = "\n\n".join(summaries)
            final_prompt = f"Please provide a concise and coherent summary of the following combined summaries:\n\n{combined_summary}"
            if "gpt" in model.lower():
                final_summary = summarize_with_openai(final_prompt, api_key, model)
            else:
                final_summary = summarize_with_anthropic(final_prompt, api_key, model)
        else:
            final_summary = summaries[0]

        logging.info("Summarization completed successfully.")
        return final_summary
    except Exception as e:
        logging.error(f"Error in summarization: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv

    load_dotenv()

    example_transcript = "This is a test transcript." * 1000  # Make it long for testing
    example_prompt_file = "prompts/prompt.txt"
    example_openai_api_key = os.getenv("OPENAI_API_KEY")
    example_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    anthropic_model = os.getenv("ANTHROPIC_MODEL")
    openai_token_limit = int(os.getenv("OPENAI_MODEL_TOKEN_LIMIT", 8000))
    anthropic_token_limit = int(os.getenv("ANTHROPIC_MODEL_TOKEN_LIMIT", 100000))

    try:
        # Test with OpenAI model
        openai_summary = summarize_transcript(
            example_transcript,
            example_prompt_file,
            example_openai_api_key,
            openai_model,
            openai_token_limit,
        )
        print(f"{openai_model} Summary:", openai_summary)

        # Test with Anthropic model
        anthropic_summary = summarize_transcript(
            example_transcript,
            example_prompt_file,
            example_anthropic_api_key,
            anthropic_model,
            anthropic_token_limit,
        )
        print(f"{anthropic_model} Summary:", anthropic_summary)
    except Exception as e:
        logging.error(f"Failed to summarize transcript: {str(e)}")
