"""
chat_server.py — Smart Mirror voice chatbot UI server.

Serves the glowing-circle chat interface and handles:
  - ASR:   browser sends WAV audio → Azure Speech SDK → transcript
  - Agent: transcript → LangChain agent → reply text
  - TTS:   reply text → Azure Speech SDK → audio bytes → browser

Run:
    uv run python main.py chat        ← recommended (starts this server)
    uv run uvicorn chat_server:app --port 8002
"""

from __future__ import annotations

import asyncio
import base64
import collections
import io
import json
import os
import queue
import re
import struct
import threading
import wave
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

# ── .env loader (same logic as agent.py) ─────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _v = _v.strip().strip('"').strip("'")
            os.environ.setdefault(_k.strip(), _v)

from agent import build_agent  # noqa: E402 (needs env loaded first)

# ── Azure Speech config ───────────────────────────────────────────────────────
_SPEECH_KEY    = os.environ["AZURE_SPEECH_KEY"]
_SPEECH_REGION = os.environ["AZURE_SPEECH_REGION"]
_SPEECH_VOICE  = os.environ.get("AZURE_SPEECH_VOICE", "en-GB-LibbyNeural")

# ─────────────────────────────────────────────────────────────────────────────
# ASR — Azure Speech SDK, push-stream so we feed WAV bytes directly
# ─────────────────────────────────────────────────────────────────────────────

def _make_speech_config() -> speechsdk.SpeechConfig:
    cfg = speechsdk.SpeechConfig(subscription=_SPEECH_KEY, region=_SPEECH_REGION)
    cfg.speech_recognition_language = "en-US"
    return cfg


def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    """Transcribe raw WAV bytes using Azure Speech SDK. Returns transcript string."""
    cfg = _make_speech_config()

    # Parse WAV header to get sample rate and bit depth
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sample_rate  = wf.getframerate()
        bits         = wf.getsampwidth() * 8
        channels     = wf.getnchannels()
        pcm_bytes    = wf.readframes(wf.getnframes())

    fmt = speechsdk.audio.AudioStreamFormat(
        samples_per_second=sample_rate,
        bits_per_sample=bits,
        channels=channels,
    )
    push_stream = speechsdk.audio.PushAudioInputStream(fmt)
    audio_cfg   = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer  = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)

    push_stream.write(pcm_bytes)
    push_stream.close()

    result = recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text.strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# TTS — Azure Speech SDK, returns WAV bytes (no speaker needed server-side)
# ─────────────────────────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """Remove markdown so TTS reads clean spoken English."""
    # Bold / italic
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    # Headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Bullet / numbered list markers → natural pause with comma or period
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Inline code / code blocks
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    # Links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Collapse excess blank lines / whitespace
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def speak_server_side(text: str) -> None:
    """Synthesize and play audio directly through the server's speakers.

    Same approach as the reference script: Azure TTS → soundfile → sounddevice.
    Bypasses all browser audio-policy issues entirely.
    """
    import io
    import sounddevice as sd
    import soundfile as sf

    cfg = _make_speech_config()
    cfg.speech_synthesis_voice_name = _SPEECH_VOICE
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    synth  = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
    result = synth.speak_text_async(text).get()
    audio_bytes = getattr(result, "audio_data", None)
    if not audio_bytes:
        return
    data, rate = sf.read(io.BytesIO(bytes(audio_bytes)), dtype="float32")
    sd.play(data, rate)
    sd.wait()


# ─────────────────────────────────────────────────────────────────────────────
# Server-side microphone capture (for Linux / Smart Mirror)
# ─────────────────────────────────────────────────────────────────────────────

_VAD_RATE           = int(os.environ.get("MAYA_VOICE_SAMPLE_RATE", "16000"))
_VAD_FRAME_MS       = int(os.environ.get("MAYA_VAD_FRAME_MS",       "30"))
_VAD_AGGRESSIVENESS = int(os.environ.get("MAYA_VAD_AGGRESSIVENESS", "1"))
_VAD_SILENCE_MS     = int(os.environ.get("MAYA_VAD_SILENCE_MS",     "700"))
_VAD_PRE_MS         = int(os.environ.get("MAYA_VAD_PRE_MS",         "400"))
_VAD_MAX_MS         = int(os.environ.get("MAYA_VAD_MAX_UTTERANCE_MS","20000"))

# Optional: set MAYA_MIC_DEVICE to a device index or name substring
# Run GET /api/audio-devices to list what's available on the mirror
_MIC_DEVICE: int | str | None = None
_raw = os.environ.get("MAYA_MIC_DEVICE", "").strip()
if _raw:
    _MIC_DEVICE = int(_raw) if _raw.isdigit() else _raw

_server_mic_stop = threading.Event()
_server_mic_rms  = 0.0   # updated every frame; read by async level-update loop


def capture_server_mic() -> bytes:
    """Record from the server microphone using webrtcvad (reference script logic).
    Returns 16-bit 16kHz mono WAV bytes. Blocks until speech ends or timeout.
    """
    global _server_mic_rms
    frame_samples  = int(_VAD_RATE * _VAD_FRAME_MS / 1000)
    silence_frames = max(1, int(_VAD_SILENCE_MS / _VAD_FRAME_MS))
    pre_frames     = max(1, int(_VAD_PRE_MS      / _VAD_FRAME_MS))
    max_frames_n   = int(_VAD_MAX_MS / _VAD_FRAME_MS)

    vad         = webrtcvad.Vad(_VAD_AGGRESSIVENESS)
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    recorded: list[np.ndarray] = []
    ring        = collections.deque(maxlen=pre_frames)
    silence_run = 0
    triggered   = False
    _server_mic_stop.clear()
    _server_mic_rms = 0.0

    def _cb(indata: np.ndarray, _frames, _time, _status):
        audio_q.put(indata.copy())

    stream_kwargs: dict = dict(
        samplerate=_VAD_RATE,
        channels=1,
        dtype="int16",
        blocksize=frame_samples,
        callback=_cb,
    )
    if _MIC_DEVICE is not None:
        stream_kwargs["device"] = _MIC_DEVICE

    with sd.InputStream(**stream_kwargs):
        total = 0
        while not _server_mic_stop.is_set():
            block = audio_q.get().reshape(-1)
            total += 1
            pcm = block.tobytes()
            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)) / 32768.0)
            _server_mic_rms = rms   # expose to async level-update loop

            try:
                is_speech = rms > 0.002 and vad.is_speech(pcm, _VAD_RATE)
            except Exception:
                is_speech = rms > 0.002

            if not triggered:
                ring.append(block)
                if is_speech:
                    triggered = True
                    recorded.extend(ring)
                    ring.clear()
                if total >= max_frames_n:
                    break
                continue

            recorded.append(block)
            silence_run = 0 if is_speech else silence_run + 1
            if silence_run >= silence_frames or total >= max_frames_n:
                break

    _server_mic_rms = 0.0
    if not recorded:
        return b""

    frames = np.concatenate(recorded, axis=0)
    # Use Python's wave module — explicit 16-bit PCM, no subtype ambiguity
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit PCM
        wf.setframerate(_VAD_RATE)
        wf.writeframes(frames.tobytes())
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Smart Mirror Chat", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(CHAT_UI)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/audio-devices")
def list_audio_devices():
    """List available audio input devices on the server.
    Use this to find the right MAYA_MIC_DEVICE index for the Smart Mirror.
    """
    devices = sd.query_devices()
    inputs  = [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"],
         "default_sr": int(d["default_samplerate"])}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]
    default_idx = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) \
                  else sd.default.device
    return {"input_devices": inputs, "current_default": default_idx,
            "MAYA_MIC_DEVICE": _MIC_DEVICE}


_LOGO_PATH     = Path(__file__).parent / "MAYA logo - Option 3.jpg"
_logo_dark_cache: bytes | None = None   # computed once, reused on every request


def _make_dark_logo() -> bytes:
    """Convert the logo to a transparent dark-mode PNG (navy → soft lavender,
    pink/teal accents kept vibrant, white background removed)."""
    import colorsys, io
    from PIL import Image

    img     = Image.open(_LOGO_PATH).convert("RGBA")
    pixels  = list(img.getdata())
    out     = []
    for r, g, b, a in pixels:
        if r > 210 and g > 210 and b > 210:        # white bg → transparent
            out.append((0, 0, 0, 0))
            continue
        lum = (0.299*r + 0.587*g + 0.114*b) / 255
        if lum < 0.40:                              # dark (navy) → light lavender
            h_hls, _, s_hls = colorsys.rgb_to_hls(r/255, g/255, b/255)
            rr, gg, bb = colorsys.hls_to_rgb(h_hls, 0.80, min(s_hls, 0.5))
            out.append((int(rr*255), int(gg*255), int(bb*255), a))
        else:                                       # pink / teal → keep vibrant
            out.append((r, g, b, a))
    img.putdata(out)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/logo")
def serve_logo_dark():
    """Serve the dark-mode (transparent background) version of the MAYA logo."""
    global _logo_dark_cache
    if _logo_dark_cache is None:
        if not _LOGO_PATH.exists():
            raise HTTPException(status_code=404, detail="Logo not found")
        _logo_dark_cache = _make_dark_logo()
    from fastapi.responses import Response
    return Response(content=_logo_dark_cache, media_type="image/png")


@app.websocket("/ws")
async def chat_ws(ws: WebSocket):
    """
    WebSocket protocol:
      Browser → Server:
        {"type": "audio",    "data": "<base64 WAV>"}  ← browser mic (laptop)
        {"type": "text",     "text": "..."}            ← typed input
        {"type": "tap"}                                ← server mic start (Smart Mirror)
        {"type": "tap_stop"}                           ← abort server mic recording

      Server → Browser:
        {"type": "listening"}                          ← server mic is now recording
        {"type": "transcript", "text": "..."}          ← what we heard
        {"type": "thinking"}                           ← agent is processing
        {"type": "response",   "text": "..."}          ← agent reply text
        {"type": "speaking"}                           ← TTS playing server-side
        {"type": "done"}                               ← TTS finished
        {"type": "error",      "text": "..."}          ← something went wrong
    """
    await ws.accept()

    agent  = build_agent()
    config = {"configurable": {"thread_id": "chat-voice-session"}}
    loop   = asyncio.get_running_loop()

    async def send(obj: dict):
        await ws.send_text(json.dumps(obj))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # ── Decode user input ──────────────────────────────────────────

            if msg["type"] == "tap_stop":
                _server_mic_stop.set()   # abort any in-progress server recording
                continue

            if msg["type"] == "tap":
                # ── Server-side mic (Smart Mirror / Linux) ─────────────────
                await send({"type": "listening"})

                # Run blocking capture in executor while streaming live RMS
                # to the browser so the level bar animates during recording
                rec_future = loop.run_in_executor(None, capture_server_mic)
                try:
                    while not rec_future.done():
                        await asyncio.sleep(0.08)
                        await send({"type": "level",
                                    "value": round(_server_mic_rms * 900, 1)})
                    wav_bytes = await rec_future
                except Exception as mic_exc:
                    await send({"type": "error", "text": f"Microphone error: {mic_exc}"})
                    continue

                if not wav_bytes:
                    await send({"type": "error", "text": "No speech detected — please speak after tapping."})
                    continue

                try:
                    transcript = await loop.run_in_executor(None, transcribe_wav_bytes, wav_bytes)
                except Exception as asr_exc:
                    await send({"type": "error", "text": f"ASR error: {asr_exc}"})
                    continue

                if not transcript:
                    await send({"type": "error", "text": "Could not recognise speech — please try again."})
                    continue

                user_text = transcript
                await send({"type": "transcript", "text": user_text})

            elif msg["type"] == "audio":
                # ── Browser mic (laptop / desktop) ────────────────────────
                wav_bytes = base64.b64decode(msg["data"])
                try:
                    transcript = await loop.run_in_executor(None, transcribe_wav_bytes, wav_bytes)
                except Exception as asr_exc:
                    await send({"type": "error", "text": f"Could not process audio: {asr_exc}"})
                    continue

                if not transcript:
                    await send({"type": "error", "text": "No speech detected — try speaking louder."})
                    continue

                user_text = transcript
                await send({"type": "transcript", "text": user_text})

            elif msg["type"] == "text":
                user_text = msg["text"].strip()
                if not user_text:
                    continue

            else:
                continue

            # ── Run agent ─────────────────────────────────────────────────
            await send({"type": "thinking"})

            try:
                ut = user_text  # capture for lambda
                result = await loop.run_in_executor(
                    None,
                    lambda: agent.invoke(
                        {"messages": [{"role": "user", "content": ut}]},
                        config=config,
                    ),
                )
                reply = result["messages"][-1].content
            except Exception as agent_exc:
                await send({"type": "error", "text": f"Agent error: {agent_exc}"})
                continue

            clean_reply = strip_markdown(reply)
            await send({"type": "response", "text": clean_reply})

            # ── Play TTS through the server's speakers (same machine as browser) ──
            try:
                await send({"type": "speaking"})
                await loop.run_in_executor(None, speak_server_side, clean_reply)
                await send({"type": "done"})
            except Exception as tts_exc:
                print(f"[TTS] {tts_exc}")
                await send({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await send({"type": "error", "text": str(exc)})
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Chat UI HTML
# ─────────────────────────────────────────────────────────────────────────────

CHAT_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Mirror — Exercise Coach</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    min-height: 100vh;
    background: radial-gradient(ellipse at center, #0d1117 0%, #000 100%);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #eee;
    overflow: hidden;
    user-select: none;
    padding-top: 120px;  /* clear the fixed header */
  }

  /* ── Header ── */
  #app-header {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 110px;
    z-index: 200;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(180deg,
      rgba(8, 10, 18, 0.92) 0%,
      rgba(8, 10, 18, 0.00) 100%);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
  }

  #app-header img {
    height: 88px;
    width: auto;
    display: block;
    filter:
      drop-shadow(0 0 28px rgba(160, 175, 230, 0.35))
      drop-shadow(0 3px  8px rgba(0,   0,   0,  0.65));
  }

  /* ── Orbit rings behind the circle ── */
  .orbit-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(99, 179, 237, 0.08);
    animation: orbit-pulse 4s ease-in-out infinite;
    pointer-events: none;
  }
  .orbit-ring:nth-child(1) { width: 340px; height: 340px; animation-delay: 0s; }
  .orbit-ring:nth-child(2) { width: 420px; height: 420px; animation-delay: 0.8s; }
  .orbit-ring:nth-child(3) { width: 500px; height: 500px; animation-delay: 1.6s; }
  @keyframes orbit-pulse {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50%       { opacity: 0.7; transform: scale(1.03); }
  }

  /* ── Main glow circle ── */
  #circle-wrap {
    position: relative;
    width: 200px; height: 200px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  #circle {
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle at 38% 35%,
      rgba(130, 200, 255, 0.45) 0%,
      rgba(60, 130, 220, 0.35) 40%,
      rgba(20, 60, 180, 0.20) 100%
    );
    box-shadow:
      0 0 30px rgba(80, 160, 255, 0.4),
      0 0 70px rgba(60, 120, 255, 0.25),
      0 0 130px rgba(40, 90, 255, 0.15),
      inset 0 0 30px rgba(255,255,255,0.06);
    border: 1.5px solid rgba(130, 190, 255, 0.3);
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    animation: idle-glow 3s ease-in-out infinite;
    display: flex; align-items: center; justify-content: center;
  }
  @keyframes idle-glow {
    0%, 100% { box-shadow: 0 0 30px rgba(80,160,255,0.4), 0 0 70px rgba(60,120,255,0.25), 0 0 130px rgba(40,90,255,0.15), inset 0 0 30px rgba(255,255,255,0.06); }
    50%       { box-shadow: 0 0 45px rgba(80,160,255,0.6), 0 0 95px rgba(60,120,255,0.38), 0 0 160px rgba(40,90,255,0.22), inset 0 0 40px rgba(255,255,255,0.09); }
  }
  #circle:active { transform: scale(0.94); }
  #circle.listening {
    animation: listening-pulse 0.8s ease-in-out infinite;
    border-color: rgba(100,255,180,0.6);
    background: radial-gradient(circle at 38% 35%,
      rgba(100,255,180,0.4) 0%,
      rgba(40,200,130,0.3) 40%,
      rgba(20,120,80,0.15) 100%
    );
  }
  @keyframes listening-pulse {
    0%, 100% { box-shadow: 0 0 35px rgba(60,220,150,0.5), 0 0 80px rgba(40,200,120,0.3); transform: scale(1); }
    50%       { box-shadow: 0 0 55px rgba(60,220,150,0.75), 0 0 120px rgba(40,200,120,0.5); transform: scale(1.04); }
  }
  #circle.thinking {
    animation: thinking-spin 1.8s linear infinite;
    border-color: rgba(255, 210, 80, 0.5);
    background: radial-gradient(circle at 38% 35%,
      rgba(255,220,100,0.35) 0%,
      rgba(200,160,40,0.25) 40%,
      rgba(120,90,20,0.12) 100%
    );
  }
  @keyframes thinking-spin {
    0%   { box-shadow: 0 0 40px rgba(220,180,60,0.55), 60px 0 80px rgba(220,180,60,0.2), -60px 0 80px rgba(220,180,60,0.2); }
    25%  { box-shadow: 0 0 40px rgba(220,180,60,0.55), 0 60px 80px rgba(220,180,60,0.2), 0 -60px 80px rgba(220,180,60,0.2); }
    50%  { box-shadow: 0 0 40px rgba(220,180,60,0.55), -60px 0 80px rgba(220,180,60,0.2), 60px 0 80px rgba(220,180,60,0.2); }
    75%  { box-shadow: 0 0 40px rgba(220,180,60,0.55), 0 -60px 80px rgba(220,180,60,0.2), 0 60px 80px rgba(220,180,60,0.2); }
    100% { box-shadow: 0 0 40px rgba(220,180,60,0.55), 60px 0 80px rgba(220,180,60,0.2), -60px 0 80px rgba(220,180,60,0.2); }
  }
  #circle.speaking {
    animation: speaking-wave 0.6s ease-in-out infinite alternate;
    border-color: rgba(200, 130, 255, 0.5);
    background: radial-gradient(circle at 38% 35%,
      rgba(200,130,255,0.38) 0%,
      rgba(150,80,220,0.28) 40%,
      rgba(80,30,150,0.14) 100%
    );
  }
  @keyframes speaking-wave {
    0%   { box-shadow: 0 0 40px rgba(180,100,255,0.5), 0 0 90px rgba(160,80,240,0.3); transform: scale(1); }
    100% { box-shadow: 0 0 65px rgba(180,100,255,0.8), 0 0 130px rgba(160,80,240,0.5); transform: scale(1.06); }
  }

  /* Mic icon inside circle */
  .mic-icon {
    width: 36px; height: 36px;
    fill: rgba(255,255,255,0.7);
    transition: opacity 0.3s;
  }
  #circle.listening .mic-icon { fill: rgba(60, 255, 180, 0.9); }
  #circle.thinking .mic-icon,
  #circle.speaking .mic-icon  { fill: rgba(255,255,255,0.5); }

  /* Hint text below circle */
  #hint {
    margin-top: 18px;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(180,200,255,0.45);
    transition: opacity 0.4s;
  }

  /* ── Audio level bar ── */
  #level-bar {
    margin-top: 14px;
    width: 160px; height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
    opacity: 0;
    transition: opacity 0.3s;
  }
  #level-bar.active { opacity: 1; }
  #level-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #3cf, #9f6fff);
    border-radius: 2px;
    transition: width 0.05s linear;
  }

  /* ── Transcript (what the user said) ── */
  #transcript {
    margin-top: 22px;
    font-size: 0.78rem;
    color: rgba(140,180,255,0.5);
    font-style: italic;
    min-height: 1.2em;
    text-align: center;
    max-width: 560px;
    padding: 0 24px;
    transition: opacity 0.4s;
  }

  /* ── Response text ── */
  #response-wrap {
    margin-top: 18px;
    width: 100%;
    max-width: 600px;
    padding: 0 28px;
    min-height: 120px;
    display: flex;
    align-items: flex-start;
    justify-content: center;
  }
  #response {
    font-size: 1.05rem;
    line-height: 1.65;
    color: rgba(220,235,255,0.92);
    text-align: center;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.6s ease, transform 0.6s ease;
  }
  #response.visible {
    opacity: 1;
    transform: translateY(0);
  }

  /* ── Text input row ── */
  #input-row {
    position: fixed;
    bottom: 28px;
    display: flex;
    gap: 10px;
    width: 100%;
    max-width: 560px;
    padding: 0 24px;
  }
  #text-input {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 10px 18px;
    color: #eee;
    font-size: 0.9rem;
    outline: none;
    transition: border-color 0.2s;
  }
  #text-input:focus { border-color: rgba(100,160,255,0.5); }
  #text-input::placeholder { color: rgba(255,255,255,0.2); }
  #send-btn {
    background: rgba(60,120,220,0.6);
    border: none; border-radius: 50%;
    width: 42px; height: 42px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.2s;
  }
  #send-btn:hover { background: rgba(80,150,255,0.75); }
  #send-btn svg { width: 18px; height: 18px; fill: #fff; }
</style>
</head>
<body>

<!-- Header -->
<header id="app-header">
  <img src="/logo" alt="MAYA">
</header>

<!-- Orbit rings -->
<div class="orbit-ring"></div>
<div class="orbit-ring"></div>
<div class="orbit-ring"></div>

<!-- Circle button -->
<div id="circle-wrap">
  <div id="circle" onclick="onCircleTap()">
    <svg class="mic-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm7 10a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.92V20H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.08A7 7 0 0 0 19 11z"/>
    </svg>
  </div>
</div>

<div id="hint">Tap to talk</div>
<div id="level-bar"><div id="level-fill"></div></div>
<div id="transcript"></div>
<div id="response-wrap"><div id="response"></div></div>

<!-- Keyboard input -->
<div id="input-row">
  <input id="text-input" type="text" placeholder="Or type here…"
         onkeydown="if(event.key==='Enter') sendText()">
  <button id="send-btn" onclick="sendText()">
    <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
  </button>
</div>

<script>
// ── WebSocket ──────────────────────────────────────────────────────────────
const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
ws.onmessage = handleMessage;
ws.onerror   = () => { setHint('Connection error — is the server running?'); setState(STATES.IDLE); };
ws.onclose   = () => { if (state !== STATES.IDLE) { showResponse('⚠ Connection closed. Please refresh.'); setState(STATES.IDLE); } };

// ── State ──────────────────────────────────────────────────────────────────
const STATES = { IDLE: 'idle', LISTENING: 'listening', THINKING: 'thinking', SPEAKING: 'speaking' };
let state = STATES.IDLE;

// ── UI refs ────────────────────────────────────────────────────────────────
const circle     = document.getElementById('circle');
const hint       = document.getElementById('hint');
const transcript = document.getElementById('transcript');
const response   = document.getElementById('response');
const levelBar   = document.getElementById('level-bar');
const levelFill  = document.getElementById('level-fill');

function setState(s) {
  state = s;
  circle.className   = s === STATES.IDLE ? '' : s;
  levelBar.className = s === STATES.LISTENING ? 'active' : '';
  const hints = {
    [STATES.IDLE]:      'Tap to talk',
    [STATES.LISTENING]: 'Listening… tap to stop',
    [STATES.THINKING]:  'Thinking…',
    [STATES.SPEAKING]:  'Speaking…',
  };
  setHint(hints[s] || '');
}
function setHint(t)      { hint.textContent = t; }
function showTranscript(t) { transcript.textContent = t ? `"${t}"` : ''; }
function showResponse(t) {
  response.classList.remove('visible');
  response.textContent = t;
  void response.offsetWidth;
  response.classList.add('visible');
}

// ── Audio capture ────────────────────────────────────────────────────────────
const SAMPLE_RATE    = 16000;
const BLOCK_SIZE     = 512;
const SILENCE_MS     = 800;
const MAX_SEC        = 20;
const BLOCK_MS       = BLOCK_SIZE / SAMPLE_RATE * 1000;          // 32 ms
const SILENCE_BLOCKS = Math.ceil(SILENCE_MS / BLOCK_MS);         // ~25
const MIN_BLOCKS     = Math.ceil(600        / BLOCK_MS);         // 600ms min before silence stops
const MAX_BLOCKS     = Math.ceil(MAX_SEC * 1000 / BLOCK_MS);

// Silence threshold — used ONLY for auto-stop, not for gating collection.
// All audio is recorded from the first tap so nothing gets clipped.
const SILENCE_THRESH = 0.003;

let gCtx         = null;
let scriptProc   = null;
let _micStream   = null;
let pcmChunks    = [];
let peakRms      = 0;      // for diagnostics
let silenceCount = 0;
let totalBlocks  = 0;

// ── AudioContext helpers ────────────────────────────────────────────────────
function _ensureCtxRunning() {
  if (!gCtx || gCtx.state === 'closed') {
    try { gCtx = new AudioContext({ sampleRate: SAMPLE_RATE }); }
    catch(_) { gCtx = new AudioContext(); }
  }
  if (gCtx.state === 'suspended') gCtx.resume().catch(() => {});
}

['click', 'keydown', 'touchstart'].forEach(ev =>
  document.addEventListener(ev, _ensureCtxRunning, { passive: true })
);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) _ensureCtxRunning();
});

// ── Mic stream — raw audio, no browser processing that can silently zero out ─
async function _getMicStream() {
  if (_micStream) {
    const track = _micStream.getAudioTracks()[0];
    if (track && track.readyState === 'live') return _micStream;
    _micStream.getTracks().forEach(t => t.stop());
    _micStream = null;
  }
  // Request raw audio first — browser processing pipelines can produce zeros
  // in createMediaStreamSource on some Chromium/Linux builds
  try {
    _micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false,
    });
  } catch(_) {
    // Fallback: let browser choose processing
    _micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  }
  return _micStream;
}

async function startRecording() {
  if (state !== STATES.IDLE) { finishRecording(); return; }

  let stream;
  try {
    stream = await _getMicStream();
  } catch(e) {
    showResponse('⚠ Microphone access denied — please allow mic access and refresh.');
    return;
  }

  pcmChunks    = [];
  peakRms      = 0;
  silenceCount = 0;
  totalBlocks  = 0;

  showTranscript('');
  showResponse('');
  response.classList.remove('visible');
  setState(STATES.LISTENING);

  _ensureCtxRunning();
  try { await gCtx.resume(); } catch(_) {}

  const src  = gCtx.createMediaStreamSource(stream);
  scriptProc = gCtx.createScriptProcessor(BLOCK_SIZE, 1, 1);

  scriptProc.onaudioprocess = (e) => {
    if (state !== STATES.LISTENING) return;
    const data = e.inputBuffer.getChannelData(0);
    totalBlocks++;

    // RMS
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    const rms = Math.sqrt(sum / data.length);
    if (rms > peakRms) peakRms = rms;

    // Level bar — collect ALL audio (no threshold gate, no clipping)
    levelFill.style.width = Math.min(100, rms * 900) + '%';
    pcmChunks.push(new Float32Array(data));

    // Silence-based auto-stop (only kicks in after minimum recording time)
    if (totalBlocks >= MIN_BLOCKS) {
      if (rms < SILENCE_THRESH) silenceCount++;
      else                      silenceCount = 0;
      if (silenceCount >= SILENCE_BLOCKS || totalBlocks >= MAX_BLOCKS) {
        finishRecording();
      }
    }
  };

  src.connect(scriptProc);
  scriptProc.connect(gCtx.destination);
}

function finishRecording() {
  if (state !== STATES.LISTENING) return;
  levelFill.style.width = '0%';
  if (scriptProc) { scriptProc.disconnect(); scriptProc = null; }
  // Do NOT stop _micStream — keep it alive so the browser never asks for
  // permission again within this session.
  processAndSend();
}

function processAndSend() {
  setState(STATES.THINKING);

  if (pcmChunks.length === 0) {
    showResponse(`⚠ No audio captured (peak RMS: ${peakRms.toFixed(5)}). Check mic permissions.`);
    setState(STATES.IDLE);
    return;
  }

  // Trim leading silence (first 200ms of silence before any signal)
  const trimBlocks = Math.ceil(200 / BLOCK_MS);
  let firstSignal  = 0;
  for (let i = 0; i < Math.min(trimBlocks * 4, pcmChunks.length); i++) {
    let s = 0;
    for (let j = 0; j < pcmChunks[i].length; j++) s += pcmChunks[i][j] * pcmChunks[i][j];
    if (Math.sqrt(s / pcmChunks[i].length) > 0.0005) { firstSignal = Math.max(0, i - 2); break; }
  }
  if (firstSignal > 0) pcmChunks = pcmChunks.slice(firstSignal);

  // Concatenate captured Float32 chunks
  const totalLen = pcmChunks.reduce((s, c) => s + c.length, 0);
  const pcmF32   = new Float32Array(totalLen);
  let off = 0;
  for (const c of pcmChunks) { pcmF32.set(c, off); off += c.length; }

  // Resample to 16kHz if the AudioContext ran at a different rate
  const actualRate = gCtx ? gCtx.sampleRate : SAMPLE_RATE;
  let final32 = pcmF32;
  if (actualRate !== SAMPLE_RATE) {
    const ratio  = actualRate / SAMPLE_RATE;
    const outLen = Math.floor(pcmF32.length / ratio);
    final32      = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) final32[i] = pcmF32[Math.floor(i * ratio)];
  }

  // Float32 → Int16
  const pcm16 = new Int16Array(final32.length);
  for (let i = 0; i < final32.length; i++) {
    const v = Math.max(-1, Math.min(1, final32[i]));
    pcm16[i] = v < 0 ? v * 0x8000 : v * 0x7FFF;
  }

  // Build WAV
  const wavB64 = buildWav(pcm16, SAMPLE_RATE);

  if (ws.readyState !== WebSocket.OPEN) {
    showResponse('⚠ Connection lost — please refresh.');
    setState(STATES.IDLE);
    return;
  }
  ws.send(JSON.stringify({ type: 'audio', data: wavB64 }));
}

function buildWav(pcm16, rate) {
  const dataLen = pcm16.byteLength;
  const buf  = new ArrayBuffer(44 + dataLen);
  const v    = new DataView(buf);
  const s    = (o, str) => { for (let i = 0; i < str.length; i++) v.setUint8(o + i, str.charCodeAt(i)); };
  s(0, 'RIFF'); v.setUint32(4, 36 + dataLen, true);
  s(8, 'WAVE'); s(12, 'fmt ');
  v.setUint32(16, 16, true);        // PCM chunk size
  v.setUint16(20, 1,  true);        // format: PCM
  v.setUint16(22, 1,  true);        // mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);  // byte rate
  v.setUint16(32, 2,  true);        // block align
  v.setUint16(34, 16, true);        // bits per sample
  s(36, 'data'); v.setUint32(40, dataLen, true);
  new Int16Array(buf, 44).set(pcm16);
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

// ── WebSocket message handler ──────────────────────────────────────────────
function handleMessage(evt) {
  const msg = JSON.parse(evt.data);
  switch (msg.type) {
    case 'listening':  setState(STATES.LISTENING); break;
    case 'level':      levelFill.style.width = Math.min(100, msg.value) + '%'; break;
    case 'transcript': showTranscript(msg.text); levelFill.style.width = '0%'; break;
    case 'thinking':   setState(STATES.THINKING); break;
    case 'response':   showResponse(msg.text); setState(STATES.IDLE); break;
    case 'speaking':   setState(STATES.SPEAKING); break;
    case 'done':       setState(STATES.IDLE); break;
    case 'error':      showResponse('⚠ ' + msg.text); setState(STATES.IDLE); break;
  }
}

// ── Mic mode — auto-detect Linux, override with ?mic=server or ?mic=browser ─
// On Linux (Smart Mirror / Debian) Chromium's Web Audio API produces zeros
// for microphone input, so we use the server's sounddevice instead.
const _params   = new URLSearchParams(location.search);
const _isLinux  = /linux/i.test(navigator.userAgent) && !/android/i.test(navigator.userAgent);
const SERVER_MIC = _params.get('mic') === 'server'  ? true
                 : _params.get('mic') === 'browser' ? false
                 : _isLinux;   // auto: Linux → server mic, everything else → browser mic

// Level bar always visible — server streams RMS updates in server-mic mode

// Small badge (bottom-right) shows which mode is active
const _badge = document.createElement('div');
_badge.id    = 'mic-badge';
_badge.textContent = SERVER_MIC ? '🎙 Server mic' : '🎙 Browser mic';
_badge.style.cssText =
  'position:fixed;bottom:76px;right:16px;font-size:0.62rem;'
  + 'color:rgba(120,180,255,0.35);letter-spacing:0.07em;cursor:pointer;'
  + 'text-transform:uppercase;';
_badge.title = 'Click to switch mic mode';
_badge.onclick = () => {
  const url = new URL(location.href);
  url.searchParams.set('mic', SERVER_MIC ? 'browser' : 'server');
  location.href = url.toString();
};
document.body.appendChild(_badge);

// ── Circle tap ─────────────────────────────────────────────────────────────
function onCircleTap() {
  if (SERVER_MIC) {
    if (state === STATES.IDLE) {
      // Tell the server to start recording from its own microphone
      ws.send(JSON.stringify({ type: 'tap' }));
      setState(STATES.THINKING);   // waiting for server to start
    } else if (state === STATES.LISTENING) {
      // Let the user abort server recording
      ws.send(JSON.stringify({ type: 'tap_stop' }));
      setState(STATES.IDLE);
    }
    return;
  }
  // Browser mic mode (laptop / desktop)
  _ensureCtxRunning();
  if      (state === STATES.IDLE)      startRecording();
  else if (state === STATES.LISTENING) finishRecording();
}

// ── Text send ──────────────────────────────────────────────────────────────
function sendText() {
  const inp = document.getElementById('text-input');
  const txt = inp.value.trim();
  if (!txt || state !== STATES.IDLE) return;
  _ensureCtxRunning();   // create+unlock AudioContext during this user gesture
  inp.value = '';
  showTranscript('');
  showResponse('');
  response.classList.remove('visible');
  setState(STATES.THINKING);
  ws.send(JSON.stringify({ type: 'text', text: txt }));
}
</script>
</body>
</html>"""
