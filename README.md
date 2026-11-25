# NCERT Class 8 Science RAG chatbot

Local RAG: PyMuPDF + bge-small-en-v1.5 + ChromaDB + Groq (streaming) + Chainlit.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then set GROQ_API_KEY and GROQ_MODEL
# put the PDF at data/science_class8.pdf
python ingest.py            # re-running is a no-op unless the PDF changes
chainlit run app.py
```

- `SCORE_THRESHOLD` in `.env`: below it the bot says "not found" and skips the LLM.
- Chapter/section detection is font-size based. Check the per-chapter counts printed by `ingest.py`;
  if headings are missed, text still falls back to 300-word chunks with 50-word overlap.
- "Page" is the PDF page index, not the printed page number.
