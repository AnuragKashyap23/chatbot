import sys
import os
import uuid
import uvicorn
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from nemoguardrails import LLMRails, RailsConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "config"))
from actions import mask_pii


class NemoSafeChatOllama(ChatOllama):
    """ChatOllama subclass that absorbs extra kwargs NeMo injects at invoke-time.

    NeMo Guardrails passes 'temperature', 'streaming', etc. as direct kwargs
    to the LLM.  The Ollama REST API expects those inside the 'options' dict,
    not as top-level params.  This class intercepts _chat_params and moves
    them into options so the call to ollama.AsyncClient.chat() succeeds.
    """

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

# ── NeMo Guardrails Pipeline ──
# 1. Load rails config (config.yml + rails.co + actions.py)
config = RailsConfig.from_path("./config")

# 2. Create ChatOllama with our NeMo-safe wrapper
llm = NemoSafeChatOllama(
    model="mistral",
    base_url="http://localhost:11434",
    num_predict=50,
)

# 3. Wire the LLM into the NeMo Guardrails engine.
#    The full NeMo pipeline: User → Input Rails → LLM → Output Rails → Response
rails = LLMRails(config, llm=llm)

sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    # ── Step 0: PII masking (runs before NeMo) ──
    # If the message contains personal info, mask it and respond immediately.
    # The raw PII never reaches the LLM.
    masked_text, pii_labels = mask_pii(request.message)

    if pii_labels:
        bot_message = (
            "I detected sensitive personal information in your message "
            f"({', '.join(pii_labels)}). Here is the masked version:\n\n"
            f"{masked_text}\n\n"
            "Please do not share personal information in chat."
        )
        sessions[session_id].append({"role": "user", "content": "[message contained PII]"})
        sessions[session_id].append({"role": "assistant", "content": bot_message})
        return ChatResponse(response=bot_message, session_id=session_id)

    # ── Step 1-6: NeMo Guardrails pipeline ──
    sessions[session_id].append({
        "role": "user",
        "content": request.message,
    })

    try:
        # NeMo handles the full pipeline:
        #   1-2. Abusive language check (rule-based + LLM)
        #   3-4. Prompt injection check (rule-based + LLM)
        #   5-6. SQL injection check (rule-based + LLM)
        #   7.   If all pass → LLM rewrites the text
        response = await rails.generate_async(
            messages=sessions[session_id]
        )

        bot_message = response.get("content", "I'm sorry, I couldn't process your request.")

        # ── Output-side PII masking (safety net) ──
        # If the LLM's response contains PII (e.g. it converted number
        # words to digits), mask it before sending to the user.
        out_masked, out_labels = mask_pii(bot_message)
        if out_labels:
            bot_message = out_masked

        sessions[session_id].append({
            "role": "assistant",
            "content": bot_message,
        })

        return ChatResponse(response=bot_message, session_id=session_id)

    except Exception as e:
        sessions[session_id].pop()
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/api/reset")
async def reset_session(session_id: Optional[str] = None):
    if session_id and session_id in sessions:
        del sessions[session_id]
    return {"status": "ok", "message": "Session reset successfully"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model": "mistral (ollama)", "guardrails": "nemo"}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
