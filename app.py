import sys
import os
import json
import uuid
import logging
import uvicorn
from typing import Optional

import ollama
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "config"))
from actions import (
    mask_pii,
    check_abusive_words,
    check_prompt_injection,
    check_sql_injection,
    COMBINED_MODERATION_PROMPT,
)

log = logging.getLogger(__name__)

app = FastAPI(title="GuardBot - AI Chatbot with NeMo Guardrails")

ollama_client = ollama.AsyncClient(host="http://localhost:11434")

SYSTEM_PROMPT = (
    "You are a text rewriter. Your ONLY job is to rewrite the user's input "
    "in proper, grammatically correct, PROFESSIONAL English. Follow these rules strictly:\n\n"
    "- Output ONLY the rewritten sentence. Nothing else. No explanations, no greetings, no extra words.\n"
    "- ALWAYS use a professional and formal tone, regardless of how the user writes.\n"
    "- If the input is in Hindi (Devanagari or transliterated), translate it to professional English.\n"
    "- Do NOT answer questions. Do NOT provide information. Just rewrite/translate the text.\n"
    "- Keep the output under 50 tokens unless the input itself exceeds 50 tokens."
)

BLOCK_MESSAGES = {
    "abusive_regex": (
        "I'm sorry, but I detected abusive or offensive language in your message. "
        "Please refrain from using such words. Could you please rephrase your message?"
    ),
    "prompt_injection_regex": (
        "I'm sorry, but your message appears to contain an attempt to manipulate "
        "my instructions. I cannot process this request. Please ask a genuine question."
    ),
    "sql_injection_regex": (
        "I'm sorry, but your message appears to contain potentially harmful code or "
        "SQL injection patterns. I cannot process this request for security reasons."
    ),
    "abusive_llm": (
        "Your message appears to contain inappropriate or offensive content. "
        "Please communicate respectfully."
    ),
    "prompt_injection_llm": (
        "Your message was flagged as an attempt to manipulate my instructions. "
        "I cannot process this request. Please ask a genuine question."
    ),
    "sql_injection_llm": (
        "Your message was flagged as containing a potential database attack pattern. "
        "I cannot process this request for security reasons."
    ),
}

sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


async def run_guardrails(message: str) -> str | None:
    """Run all guardrail checks. Returns a block message key or None if safe."""
    ctx = {"user_message": message}

    if await check_abusive_words(context=ctx):
        return "abusive_regex"
    if await check_prompt_injection(context=ctx):
        return "prompt_injection_regex"
    if await check_sql_injection(context=ctx):
        return "sql_injection_regex"

    try:
        prompt = COMBINED_MODERATION_PROMPT.format(message=message)
        resp = await ollama_client.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 10},
        )
        raw = resp["message"]["content"].strip().lower().replace(" ", "")
        parts = [p.strip() for p in raw.split(",")]

        if len(parts) > 0 and parts[0].startswith("yes"):
            return "abusive_llm"
        if len(parts) > 1 and parts[1].startswith("yes"):
            return "prompt_injection_llm"
        if len(parts) > 2 and parts[2].startswith("yes"):
            return "sql_injection_llm"
    except Exception as e:
        log.warning(f"LLM moderation failed, allowing through: {e}")

    return None


async def stream_rewrite(message: str, session_id: str):
    """Stream the LLM rewrite token-by-token via SSE."""
    yield _sse({"type": "start", "session_id": session_id})

    full_text = ""
    try:
        response = await ollama_client.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            stream=True,
            options={"num_predict": 50},
        )

        async for chunk in response:
            token = chunk["message"]["content"]
            full_text += token
            yield _sse({"type": "token", "content": token})

    except Exception as e:
        log.error(f"Ollama streaming error: {e}")
        yield _sse({"type": "error", "content": "Error generating response."})
        yield _sse({"type": "done"})
        return

    masked, labels = mask_pii(full_text)
    if labels:
        yield _sse({"type": "replace", "content": masked})

    sessions.setdefault(session_id, [])
    final_text = masked if labels else full_text
    sessions[session_id].append({"role": "user", "content": message})
    sessions[session_id].append({"role": "assistant", "content": final_text})

    yield _sse({"type": "done"})


@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    # ── Step 0: PII masking ──
    masked_text, pii_labels = mask_pii(request.message)
    if pii_labels:
        bot_message = (
            "I detected sensitive personal information in your message "
            f"({', '.join(pii_labels)}). Here is the masked version:\n\n"
            f"{masked_text}\n\n"
            "Please do not share personal information in chat."
        )
        sessions.setdefault(session_id, [])
        sessions[session_id].append({"role": "user", "content": "[message contained PII]"})
        sessions[session_id].append({"role": "assistant", "content": bot_message})

        async def pii_stream():
            yield _sse({"type": "blocked", "content": bot_message, "session_id": session_id})
            yield _sse({"type": "done"})

        return StreamingResponse(pii_stream(), media_type="text/event-stream")

    # ── Step 1: Guardrails (regex + 1 LLM call) ──
    block_key = await run_guardrails(request.message)
    if block_key:
        bot_message = BLOCK_MESSAGES[block_key]
        sessions.setdefault(session_id, [])
        sessions[session_id].append({"role": "user", "content": request.message})
        sessions[session_id].append({"role": "assistant", "content": bot_message})

        async def blocked_stream():
            yield _sse({"type": "blocked", "content": bot_message, "session_id": session_id})
            yield _sse({"type": "done"})

        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    # ── Step 2: Stream rewrite directly from Ollama ──
    return StreamingResponse(
        stream_rewrite(request.message, session_id),
        media_type="text/event-stream",
    )


@app.post("/api/reset")
async def reset_session(session_id: Optional[str] = None):
    if session_id and session_id in sessions:
        del sessions[session_id]
    return {"status": "ok", "message": "Session reset successfully"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model": "mistral (ollama)", "guardrails": "direct"}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
