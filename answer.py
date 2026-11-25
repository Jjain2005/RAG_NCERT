from groq import AsyncGroq

from config import GROQ_API_KEY, GROQ_MODEL, SCORE_THRESHOLD

NOT_FOUND = (
    "I couldn't find this topic in the Class 8 Science book. "
    "Try rephrasing, or ask about a topic from the textbook."
)

# P2 Fix: Adaptive system prompt — format scales with question complexity
SYSTEM = """You are a friendly Class 8 science tutor for NCERT students.
Answer ONLY from the textbook context provided in the user message.
If the context does not contain enough information, say clearly: "This specific detail is not covered in the provided context."
Do NOT use outside knowledge — only what is in the context.

Adapt your format to the question type:

For FACTUAL questions (e.g. "What is X?"):
  **Answer**: 2–3 clear sentences.
  **Key points**: 2–4 bullets.
  **Source**: chapter, section and page from the context labels.

For EXPLANATION questions (e.g. "Why/How does X work?"):
  **Short answer**: 1–2 sentences.
  **Explanation**: step-by-step, in simple language.
  **Example**: use the book's own example. If none exists, write one and label it "Added example, not from the textbook."
  **Key points**: 3–5 bullets.
  **Source**: chapter, section and page from the context labels.

For COMPARISON questions (e.g. "Difference between X and Y"):
  Use a markdown comparison table with Property | X | Y columns.
  **Key points**: 3–4 bullets summarising the main differences.
  **Source**: chapter, section and page from the context labels.

Rules:
- Write in simple English suitable for a 13-year-old.
- Never invent a source. Copy the [Chapter | Section | Page] label exactly from the context.
- Keep answers concise — do not pad with filler sentences."""


def found(hits):
    return bool(hits) and hits[0]["score"] >= SCORE_THRESHOLD


def _label(c):
    m = c["meta"]
    return (
        f"[Chapter {m['chapter_no']}: {m['chapter_title']} "
        f"| Section: {m['section'] or '-'} | Page {m['page']}]"
    )


# P3 Fix: Reuse AsyncGroq client across requests (connection pooling)
_groq_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    return _groq_client


async def stream_answer(question, hits, neighbours, history):
    """Yield answer tokens. history = [(question, answer), ...]."""
    if not found(hits):
        yield NOT_FOUND          # below threshold: no LLM call
        return
    if not GROQ_API_KEY or not GROQ_MODEL:
        yield (
            "⚠️ **Groq API Key Missing**: Please set your `GROQ_API_KEY` "
            "in the `.env` file.\n\nRetrieved textbook context is shown "
            "in the sources drawer above."
        )
        return

    # Build deduplicated context parts
    seen, parts = set(), []
    for c in hits + neighbours:
        if c["meta"]["idx"] not in seen:
            seen.add(c["meta"]["idx"])
            parts.append(f"{_label(c)}\n{c['text']}")

    messages = [{"role": "system", "content": SYSTEM}]

    # P3 Fix: Truncate history answers to 200 chars to keep prompt lean
    for q, a in history[-3:]:
        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a[:200] + "…" if len(a) > 200 else a},
        ]

    messages.append({
        "role": "user",
        "content": (
            "Textbook context:\n\n"
            + "\n\n".join(parts)
            + f"\n\nQuestion: {question}"
        ),
    })

    try:
        stream = await _get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.15,
            max_tokens=1024,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        err = str(e)
        if "404" in err or "model_not_found" in err:
            yield (
                f"⚠️ **Model not found**: `{GROQ_MODEL}` is not available on your Groq account.\n\n"
                "Check your plan at https://console.groq.com/docs/models and update `GROQ_MODEL` in `.env`."
            )
        elif "401" in err or "auth" in err.lower():
            yield "⚠️ **Invalid API key**: Check `GROQ_API_KEY` in your `.env` file."
        else:
            yield f"⚠️ **LLM error**: {err}"
