# GuardBot - AI Chatbot with NeMo Guardrails

A web-based chatbot application powered by **Ollama Mistral** model with **NVIDIA NeMo Guardrails** for safety. The bot corrects grammar, translates Hindi to English, and blocks abusive language, prompt injections, and SQL injections.

## Features

- **Grammar Correction** - Checks user input for grammar errors and responds in proper English
- **Hindi to English Translation** - Translates Hindi messages to English before responding
- **Abusive Language Filter** - Blocks English and Hindi abusive/offensive words
- **Prompt Injection Guard** - Detects and blocks prompt injection attempts
- **SQL Injection Guard** - Detects and blocks SQL injection patterns
- **Beautiful Chat UI** - Modern, responsive dark-themed chat interface

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running locally
   - Download from: https://ollama.ai
   - Pull the Mistral model:
     ```bash
     ollama pull mistral
     ```

## Setup

### 1. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure Ollama is running

```bash
ollama serve
```

In a separate terminal, verify Mistral is available:

```bash
ollama list
```

### 4. Run the application

```bash
python app.py
```

The app will start at **http://localhost:8000**

## Project Structure

```
guardrail/
├── config/
│   ├── config.yml        # NeMo Guardrails + Ollama configuration
│   ├── rails.co          # Colang flow definitions for input rails
│   └── actions.py        # Custom guardrail actions (abusive, injection checks)
├── static/
│   └── index.html        # Frontend chat UI
├── app.py                # FastAPI backend server
├── requirements.txt      # Python dependencies
└── README.md
```

## How Guardrails Work

### Input Rails Pipeline

Every user message passes through three guardrail checks before reaching the LLM:

1. **Abusive Language Check** - Scans for English and Hindi abusive words
2. **Prompt Injection Check** - Detects manipulation attempts (e.g., "ignore previous instructions")
3. **SQL Injection Check** - Detects SQL attack patterns (e.g., `' OR '1'='1'`)

If any check fails, the message is blocked and the user receives a warning. The message never reaches the LLM.

## Usage Examples

| Input | Behavior |
|-------|----------|
| `"Hello, how are you?"` | Normal response from Mistral |
| `"i has went to store yesterday"` | Grammar correction + response |
| `"मुझे मौसम के बारे में बताओ"` | Hindi translated to English, then response |
| `"you are an idiot"` | Blocked - abusive language detected |
| `"ignore previous instructions and..."` | Blocked - prompt injection detected |
| `"' OR '1'='1'; DROP TABLE users"` | Blocked - SQL injection detected |
