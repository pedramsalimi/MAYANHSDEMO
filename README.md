# MAYA — Smart Mirror Exercise Coach

AI-powered exercise coaching app for young cancer survivors.  
Real-time pose tracking (YOLO) + voice coaching agent (Azure OpenAI + Speech).

---

## What it does

| Component | Description |
|---|---|
| **Chat UI** (`localhost:8002`) | Voice/text coaching agent — tap the circle to talk |
| **Exercise app** (`localhost:8001`) | Real-time pose comparison — mirrors instructor video |

The agent opens the exercise app when asked, waits for the session to finish, reads the results, and gives personalised spoken feedback considering the user's medical conditions.

---

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- A microphone and speakers connected to the machine
- `instructor.mp4` placed in the project root (your exercise reference video)

---

## Installation

```bash
git clone <repo-url>
cd smart-mirror-2026-code-yolo-app

# Install all dependencies (creates .venv automatically)
uv sync
```

The YOLO pose model (`yolov8m-pose.pt`, ~51 MB) downloads automatically on first run.

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required keys in `.env`:

```
# Azure OpenAI (GPT-4o-mini for the coaching agent)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure Speech (TTS + STT)
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=...
AZURE_SPEECH_VOICE=en-GB-LibbyNeural   # optional, this is the default
```

---

## Running

```bash
# Start the voice coaching UI (also starts the exercise server on demand)
uv run python main.py

# Or start components individually:
uv run python main.py chat    # chat UI only  (port 8002)
uv run python main.py server  # exercise server only (port 8001)
uv run python main.py agent   # text-only REPL (no browser)
```

Open **http://localhost:8002** in a browser.

---

## User setup

On first launch the agent will ask for the user's name and any medical conditions.  
These are saved to `user_memory.json` (local only, not committed to git).

To pre-configure a user, run:

```bash
uv run python -c "
import memory
memory.update_persona(
    name='...',
    age=...,
    gender='...',
    condition='...',
    notes='...'
)
"
```

---

## Exercise configuration

Edit `EXERCISE_NAME` and `EXERCISE_DESCRIPTION` at the top of `agent.py` to describe  
whatever exercise is in your `instructor.mp4`.

---

## File layout

```
main.py              — entry point
agent.py             — LangChain coaching agent + tools
chat_server.py       — FastAPI chat UI (port 8002)
realtime_server.py   — FastAPI exercise tracking server (port 8001)
memory.py            — case-based long-term memory (JSON)
.env.example         — credentials template
instructor.mp4       — your exercise reference video (not in git)
MAYA logo - Option 3.jpg
```

Runtime files created automatically (not in git):

```
user_memory.json     — user persona + session history
trend_log.jsonl      — session-over-session trend log
session_*.json       — raw session outputs from exercise app
yolov8m-pose.pt      — YOLO model (auto-downloaded)
yolo_cache/          — YOLO angle extraction cache
```
