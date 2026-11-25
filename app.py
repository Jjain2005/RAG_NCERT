from typing import Optional

import chainlit as cl
from chainlit.types import ThreadDict
from starlette.datastructures import Headers

from answer import found, stream_answer
from data_layer import SQLiteDataLayer
from retrieve import retrieve

GREETINGS = {"hi", "hey", "hello", "hii", "helo", "yo", "sup", "howdy", "greetings"}

GREETING_REPLY = (
    "👋 Hi there! I'm your **NCERT Class 8 Science** tutor.\n\n"
    "Ask me anything from the textbook — like:\n"
    "- *What is combustion?*\n"
    "- *Explain the laws of reflection of light.*\n"
    "- *Difference between plant cell and animal cell.*\n\n"
    "What would you like to learn today? 📚"
)


@cl.data_layer
def get_data_layer():
    return SQLiteDataLayer()


@cl.header_auth_callback
async def auth_header(headers: Headers) -> Optional[cl.User]:
    """Auto-authenticate local user so thread history & left sidebar work seamlessly."""
    return cl.User(identifier="student", metadata={"role": "student"})


@cl.set_starters
async def starters(*_, **__):  # accepts whatever args this Chainlit version passes
    return [
        cl.Starter(label="Pressure", message="What is pressure and why does a sharp knife cut better than a blunt one?"),
        cl.Starter(label="Plant vs animal cell", message="What is the difference between a plant cell and an animal cell?"),
        cl.Starter(label="Reflection of light", message="Explain the laws of reflection of light."),
        cl.Starter(label="Metals vs non-metals", message="How do metals differ from non-metals?"),
    ]


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])


@cl.on_chat_resume
async def resume(thread: ThreadDict):
    """Restore (question, answer) history when clicking a past chat in the sidebar."""
    history: list[tuple[str, str]] = []
    pending_q: Optional[str] = None
    for step in thread.get("steps", []):
        stype = step.get("type")
        text = (step.get("output") or "").strip()
        if stype == "user_message" and text:
            pending_q = text
        elif stype == "assistant_message" and text and pending_q is not None:
            history.append((pending_q, text))
            pending_q = None
    cl.user_session.set("history", history[-3:])


@cl.on_message
async def main(msg: cl.Message):
    history = cl.user_session.get("history") or []
    q = msg.content.strip()

    # Handle greetings without hitting retrieval + LLM
    if q.lower().rstrip("!?.") in GREETINGS:
        await cl.Message(content=GREETING_REPLY).send()
        return

    # very short follow-ups ("why?") borrow the previous question for retrieval
    search_q = q if len(q.split()) >= 6 or not history else f"{history[-1][0]} {q}"
    hits, neighbours = await cl.make_async(retrieve)(search_q)

    if found(hits):
        async with cl.Step(name="Source from the book") as step:
            step.output = "\n\n---\n\n".join(
                f"**Chapter {h['meta']['chapter_no']}: {h['meta']['chapter_title']}**"
                f" | {h['meta']['section'] or '-'} | page {h['meta']['page']}"
                f" | {h['meta']['chunk_type']} | score {h['score']:.2f}\n\n{h['text']}"
                for h in hits)

    out = cl.Message(content="")
    async for token in stream_answer(q, hits, neighbours, history):
        await out.stream_token(token)
    await out.send()

    # Only keep valid Q&A turns in follow-up history
    if found(hits) and not out.content.startswith("⚠️"):
        history.append((q, out.content))
        del history[:-3]  # keep last 3 turns
        cl.user_session.set("history", history)
