import os
from openai import OpenAI

def load_prompt(prompt_file):
    with open(prompt_file, 'r') as file:
        prompt = file.read()
    return prompt

def summarize_transcript(transcript, prompt_file, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    prompt = load_prompt(prompt_file).replace("{{TRANSCRIPT}}", transcript)

    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # Example usage
    example_transcript = "This is a test transcript."
    example_prompt_file = "prompts/summary_prompt.txt"
    example_openai_api_key = os.getenv('OPENAI_API_KEY')

    summary = summarize_transcript(example_transcript, example_prompt_file, example_openai_api_key)
    print(summary)