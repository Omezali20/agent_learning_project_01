"""
Central place for configuration and secrets.

Why this file exists: hardcoding API keys directly in code (like the fake keys in
../LLM Intro.ipynb) means they end up in git history, notebook outputs, and anywhere
the file is shared. Instead, secrets live in a single .env file at the project root
(gitignored) and every other module reads them from here - one source of truth.
"""

import os

from dotenv import load_dotenv

# load_dotenv() searches the current directory and walks upward through parent
# directories until it finds a .env file. Our .env lives at the repo root, one
# level above this file, so this works whether you run scripts from the repo
# root or from inside this project folder.
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is missing. Add it to the .env file at the project root, e.g.\n"
        '  GROQ_API_KEY = "your-key-here"\n'
        "Get a free key at https://console.groq.com/keys"
    )

if not TAVILY_API_KEY:
    raise EnvironmentError(
        "TAVILY_API_KEY is missing. Add it to the .env file at the project root, e.g.\n"
        '  TAVILY_API_KEY = "your-key-here"\n'
        "Get a free key at https://app.tavily.com"
    )

# openai/gpt-oss-120b: OpenAI's open-weight model, hosted on Groq's fast
# inference, with solid tool-calling support. (Groq's catalog changes over
# time - if this model ever disappears, run list_groq_models.py in this
# folder to see what's currently available and swap the name here.)
# temperature=0 makes tool-use decisions as deterministic and repeatable as
# possible, which matters most while you're learning how the agent reasons -
# randomness would make its behavior harder to follow.
MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0
