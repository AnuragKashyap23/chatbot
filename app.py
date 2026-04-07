import sys
import os
import json
import uuid
import logging
import uvicorn
from typing import Any, Optional

import ollama
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from nemoguardrails import LLMRails, RailsConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "config"))
from actions import mask_pii

log = logging.getLogger(__name__)

GUARDRAILS_PASSED = "GUARDRAILS_PASSED"


class NemoSafeChatOllama(ChatOllama):
    """ChatOllama subclass that moves NeMo-injected kwargs into Ollama's options dict."""

    def _chat_params(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        temp_override = kwargs.pop("temperature", None)
        kwargs.pop("streaming", None)
        params = super()._chat_params(messages, stop, **kwargs)
        if temp_override is not None and "options" in params:
            params["options"]["temperature"] = temp_override
        return params


app = FastAPI(title="GuardBot - AI Chatbot with NeMo Guardrails")

# ── NeMo Guardrails (input rails only) ──
config = RailsConfig.from_path("./config")
nemo_llm = NemoSafeChatOllama(
    model="mistral",
    base_url="http://localhost:11434",
    num_predict=10,
)
rails = LLMRails(config, llm=nemo_llm)

# ── Direct Ollama client (streaming rewrite) ──
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

sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


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

    # ── Step 0: PII masking (pre-NeMo) ──
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

    # ── Step 1: NeMo input rails (guardrail checks via rails.co) ──
    # NeMo runs: regex checks → combined LLM moderation → signal safe passage
    # If blocked → returns block message from rails.co
    # If safe → returns "GUARDRAILS_PASSED" (our signal subflow)
    try:
        nemo_response = await rails.generate_async(
            messages=[{"role": "user", "content": request.message}]
        )
        bot_message = nemo_response.get("content", "")
    except Exception as e:
        log.error(f"NeMo guardrails error: {e}")
        bot_message = GUARDRAILS_PASSED

    if bot_message != GUARDRAILS_PASSED:
        # Blocked by a guardrail — return the block message
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
    return {"status": "healthy", "model": "mistral (ollama)", "guardrails": "nemo + streaming"}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
