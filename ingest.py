"""Parse the NCERT PDF, chunk it, embed it, store it in Chroma. Idempotent."""
import hashlib
import re
from collections import Counter
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from config import BOOK_ID, CHROMA_DIR, COLLECTION, EMBED_MODEL, PDF_PATH

CHUNK_WORDS, OVERLAP, MIN_SECTION = 300, 50, 40


def page_lines(page):
    """One dict per text line: text, font size, bold, vertical position."""
    out = []
    for b in page.get_text("dict", sort=True)["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans:
                continue
            out.append({
                "text": " ".join("".join(s["text"] for s in spans).split()),
                "size": round(max(s["size"] for s in spans), 1),
                "bold": all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans),
                "y0": l["bbox"][1],
                "y1": l["bbox"][3],
            })
    return out


def strip_headers(pages, heights):
    """Drop page numbers and running headers/footers (margin lines that repeat)."""
    norm = lambda t: re.sub(r"\d+", "#", t.lower())
    edge = lambda l, h: l["y0"] < 0.1 * h or l["y1"] > 0.9 * h
    counts = Counter(norm(l["text"]) for p, h in zip(pages, heights) for l in p if edge(l, h))
    return [
        [l for l in p if not (edge(l, h) and (l["text"].isdigit()
                                              or counts[norm(l["text"])] >= 3
                                              or "reprint" in l["text"].lower()))]
        for p, h in zip(pages, heights)
    ]


def is_head(l, body):
    t = l["text"]
    if t.isdigit() or len(t.split()) > 12 or t.endswith((".", ",", ";")):
        return False
    return l["size"] >= body * 1.15 or l["bold"]


def ctype(sec):
    s = sec.lower()
    if re.search(r"\bactivity|\bexperiment", s):
        return "activity"
    if re.search(r"\bexample", s):
        return "example"
    if re.search(r"\bexercises?\b|\bquestions\b|\bpractice\b|\bassignment", s):
        return "exercise"
    return "concept" if sec else "other"


def split(words):
    """~300-word pieces with 50-word overlap. words = [(word, page)]."""
    if not words:
        return []
    if len(words) <= CHUNK_WORDS:
        return [words]
    out = []
    for i in range(0, len(words), CHUNK_WORDS - OVERLAP):
        out.append(words[i:i + CHUNK_WORDS])
        if i + CHUNK_WORDS >= len(words):
            break
    return out


CHAPTER_MAP = {
    14: (1, "Crop Production and Management"),
    30: (2, "Microorganisms: Friend and Foe"),
    45: (3, "Synthetic Fibres and Plastics"),
    57: (4, "Materials: Metals and Non-Metals"),
    69: (5, "Coal and Petroleum"),
    77: (6, "Combustion and Flame"),
    90: (7, "Conservation of Plants and Animals"),
    103: (8, "Cell - Structure and Functions"),
    113: (9, "Reproduction in Animals"),
    126: (10, "Reaching the Age of Adolescence"),
    140: (11, "Force and Pressure"),
    159: (12, "Friction"),
    170: (13, "Sound"),
    185: (14, "Chemical Effects of Electric Current"),
    197: (15, "Some Natural Phenomena"),
    212: (16, "Light"),
    228: (17, "Stars and the Solar System"),
    252: (18, "Pollution of Air and Water"),
}


def parse(pdf):
    doc = fitz.open(pdf)
    pages = strip_headers([page_lines(p) for p in doc], [p.rect.height for p in doc])
    sizes = Counter()
    for p in pages:
        for l in p:
            sizes[l["size"]] += len(l["text"])
    body = sizes.most_common(1)[0][0]  # most common font size = body text

    chunks, buf = [], []
    chapter, sec, prev_head = (0, "Front matter"), "", False

    def flush():
        nonlocal buf
        if not buf:
            return
        for piece in split(buf):
            if not piece:
                continue
            chunks.append({
                "chapter_no": chapter[0], "chapter_title": chapter[1], "section": sec,
                "page": piece[0][1], "chunk_type": ctype(sec),
                "text": " ".join(w for w, _ in piece),
            })
        buf = []

    for pno, lines in enumerate(pages, 1):
        if pno in CHAPTER_MAP:
            flush()
            chapter, sec, prev_head = CHAPTER_MAP[pno], "", False

        # Also support dynamic chapter detection
        big = [l for l in lines if l["size"] >= body * 1.6]
        num, title = None, []
        for l in big:
            m = re.match(r"^(?:chapter\s+)?(\d{1,2})(?:\s*[:.\-]\s*|\s+|$)(.*)$", l["text"], re.I)
            rest = l["text"]
            if m:
                num, rest = num or m.group(1), m.group(2)
            if rest:
                title.append(rest)
        new_chapter = bool(num and title and int(num) != chapter[0])
        big_ids = {id(l) for l in big} if new_chapter else set()
        if new_chapter:
            flush()
            chapter, sec, prev_head = (int(num), " ".join(title)), "", False

        for l in lines:
            if id(l) in big_ids:
                continue
            t = l["text"]
            # Detect section heading
            sec_match = re.match(r"^(\d{1,2})\.\d+\s+(.*)", t)
            if sec_match and int(sec_match.group(1)) == chapter[0]:
                flush()
                sec, prev_head = t, True
                continue
            if is_head(l, body):
                if prev_head:                      # heading wrapped over 2 lines
                    sec += " " + t
                    continue
                if not buf or len(buf) >= MIN_SECTION:   # start a new section
                    flush()
                    sec, prev_head = t, True
                    continue
            prev_head = False
            buf += [(w, pno) for w in t.split()]
    flush()

    if any(c["chapter_no"] > 0 for c in chunks):       # drop preface/contents
        chunks = [c for c in chunks if c["chapter_no"] > 0]
    return chunks


def main():
    pdf = Path(PDF_PATH)
    if not pdf.exists():
        raise SystemExit(f"Put the NCERT PDF at {PDF_PATH}")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()

    client = chromadb.PersistentClient(CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    old = col.get(limit=1)
    if old["ids"] and old["metadatas"][0].get("pdf_hash") == digest:
        print("PDF unchanged, skipping ingest.")
        return

    client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    chunks = parse(pdf)
    model = SentenceTransformer(EMBED_MODEL)
    texts = [f"{c['chapter_title']}. {c['section']}. {c['text']}" for c in chunks]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()
    col.add(
        ids=[f"{BOOK_ID}-{i:05d}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=embs,
        metadatas=[
            {"book_id": BOOK_ID, "idx": i, "pdf_hash": digest,
             **{k: v for k, v in c.items() if k != "text"}}
            for i, c in enumerate(chunks)
        ],
    )

    print("\nChunks per chapter:")
    for (no, title), n in sorted(Counter((c["chapter_no"], c["chapter_title"]) for c in chunks).items()):
        print(f"  {no:>2}  {title[:50]:<50} {n}")
    print(f"Total: {len(chunks)}")


if __name__ == "__main__":
    main()
