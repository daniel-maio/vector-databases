import os
from dotenv import load_dotenv
from pathlib import Path

# config.py must be in root of the project
PROJECT_ROOT = Path(__file__).resolve().parent # get the project root dir

load_dotenv(PROJECT_ROOT/".env")

# --- Encoder Model ---
ENCODER_BASE_MODEL = os.getenv('ENCODER_MODEL')

# --- LLM Model ---
GROK_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---  Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "standard-dense-py"

EMBEDDING_MODEL_PATH = PROJECT_ROOT/"models/"
TRAINING_DATA_PATH = PROJECT_ROOT/"data/training_data.json"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT/"data/knowledge_base.json"