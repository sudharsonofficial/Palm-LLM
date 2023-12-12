# Import necessary libraries
from dotenv import load_dotenv, find_dotenv
import google.generativeai as palm
import os

# Function to interact with Google Palm 2 LLM
def palmllm(query):
    # Load environment variables from .env file
    load_dotenv(find_dotenv())
    # Get API key from environment variable
    api_key = os.environ['GOOGLE_API_KEY']  # put your API key here
    # Configure Google Palm API with the obtained key
    palm.configure(api_key=api_key)

    # List available models and select the first one that supports text generation
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name

    # Generate text using the selected model and specified parameters
    text = palm.generate_text(
        prompt=query,
        model=model,
        temperature=0.6,
        max_output_tokens=200,
        top_p=0.9,
        top_k=40,
        stop_sequences=['\n']
    )

    return text.result