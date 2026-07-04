import os
from dotenv import load_dotenv

from groq import Groq

from pinecone import Pinecone

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc_client = Pinecone(api_key=PINECONE_API_KEY)