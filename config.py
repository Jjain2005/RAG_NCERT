import os
from dotenv import load_dotenv

load_dotenv()

BOOK_ID = "ncert-class8-science"
PDF_PATH = "data/science_class8.pdf"
CHROMA_DIR = "data/chroma"
COLLECTION = "ncert_class8_science"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.35"))
