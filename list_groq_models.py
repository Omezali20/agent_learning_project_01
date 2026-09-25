"""
Utility: print every model currently available to your Groq API key.

Groq's hosted model catalog changes over time (models get added/retired).
If MODEL_NAME in config.py ever starts failing with a "does not exist" error,
run this script to see current options and update config.py accordingly.
"""

import requests

from config import GROQ_API_KEY

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
    timeout=10,
)
response.raise_for_status()

for model in response.json().get("data", []):
    print(model["id"])
