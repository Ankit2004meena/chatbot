import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models
models = genai.list_models()

# Print model details
for model in models:
    print(f"Model Name: {model.name}")
    print(f"Supported Methods: {model.supported_generation_methods}")
    print("-" * 50)
