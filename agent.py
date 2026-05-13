"""
agent.py — Smart Mirror coaching agent with case-based long-term memory.

Long-term stores (see memory.py):
  user_memory.json  — persona + one case per session (joint stats + feedback given)
  trend_log.jsonl   — append-only trend observations (separate from the case-base)

Run:
    uv run python main.py          # starts chat UI (recommended)
    uv run python main.py agent    # text-only REPL
"""

from __future__ import annotations

import glob
import json
import os
import socket
import subprocess
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── .env loader ───────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _v = _v.strip().strip('"').strip("'")
            os.environ.setdefault(_k.strip(), _v)

import requests
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

import memory as mem

# ── Exercise configuration ────────────────────────────────────────────────────
# Edit these to describe whatever instructor video is loaded in the app.
EXERCISE_NAME = "Shoulder and Hip Mobility Routine"
EXERCISE_DESCRIPTION = (
    "A gentle full-body mobility routine designed for cancer survivors. "
    "The participant mirrors the instructor's movements: slow shoulder raises, "
    "arm circles, hip rotations, and controlled leg lifts. "
    "The goal is a comfortable range of motion — not speed, strength, or perfection."
)

# ── Server management ─────────────────────────────────────────────────────────
SERVER_PORT      = 8001
INSTRUCTOR_VIDEO = Path(__file__).parent / "instructor.mp4"
_server_proc: subprocess.Popen | None = None


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_server() -> str:
    global _server_proc
    if not _port_in_use(SERVER_PORT):
        _server_proc = subprocess.Popen(
            ["uvicorn", "realtime_server:app",
             "--host", "0.0.0.0", "--port", str(SERVER_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(20):
            time.sleep(0.5)
            if _port_in_use(SERVER_PORT):
                break
    return f"http://localhost:{SERVER_PORT}"


def _prepare_instructor(base_url: str) -> str:
    if not INSTRUCTOR_VIDEO.exists():
        raise FileNotFoundError(f"instructor.mp4 not found at {INSTRUCTOR_VIDEO}")
    with open(INSTRUCTOR_VIDEO, "rb") as f:
        resp = requests.post(
            f"{base_url}/prepare",
            files={"ref": (INSTRUCTOR_VIDEO.name, f, "video/mp4")},
            data={"model_name": "yolov8m-pose.pt", "conf": "0.5"},
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json()["ref_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def launch_exercise_app() -> str:
    """Start the exercise tracking app with the instructor video pre-loaded.

    Call this when the user says they want to do their exercises.
    The browser opens directly to the session — the instructor video is already
    processed so the user only needs to click Start and follow along.
    The session summary is saved automatically when the video ends.
    Ask the user to come back here when done.
    """
    base_url = _start_server()
    print("\n[Agent] Preparing instructor video (cached after first run)…")
    try:
        ref_id = _prepare_instructor(base_url)
    except Exception as exc:
        return (
            f"Could not prepare the instructor video: {exc}\n"
            f"Check that instructor.mp4 exists and {base_url} is reachable."
        )
    webbrowser.open(f"{base_url}/?ref_id={ref_id}")
    return (
        f"Exercise app is open and ready at {base_url}/?ref_id={ref_id}.\n"
        "The instructor video is already loaded — just click Start and follow along.\n"
        "The session ends automatically. Come back here when you are done."
    )


@tool
def get_latest_session_results() -> str:
    """Read and return a clean summary of the most recent exercise session.

    Call this as the first step when the user says they have finished exercising.
    Returns date, day of week, joint performance table, and the session filename
    needed for save_session_to_memory.
    Does NOT return the raw per-frame records or the misleading overall form accuracy.
    """
    pattern = str(Path(__file__).parent / "session_*.json")
    files   = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return (
            "No session file found. Make sure the instructor video played all the "
            "way through — the summary is only saved when the video ends naturally."
        )

    latest = files[0]
    try:
        with open(latest) as f:
            data = json.load(f)
    except Exception as exc:
        return f"Could not read session file: {exc}"

    meta  = data.get("meta", {})
    jstat = data.get("joint_stats", {})
    now   = datetime.now()

    # Build clean joint table (no per-frame noise)
    rows = []
    for jn, s in jstat.items():
        sign = "+" if s["mean_diff"] >= 0 else ""
        rows.append(
            f"  {jn:<18}  bad={s['bad_pct']}%  "
            f"mean_diff={sign}{s['mean_diff']}°  "
            f"max_diff={s['max_abs_diff']}°"
        )
    table = "\n".join(rows)

    return (
        f"SESSION FILE : {Path(latest).name}\n"
        f"DATE         : {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})\n"
        f"FRAMES       : {meta.get('total_patient_frames','?')} patient frames sampled\n"
        f"THRESHOLD    : {meta.get('angle_threshold', 25)}° (frame is BAD when |diff| >= threshold)\n"
        f"\nJOINT PERFORMANCE\n"
        f"  {'Joint':<18}  {'Bad %':>6}  {'Mean diff':>10}  {'Max diff':>9}\n"
        f"  {'-'*56}\n"
        f"{table}\n"
        f"\nNOTE: Do NOT use overall form accuracy (ok_pct={meta.get('ok_pct','?')}%) in "
        f"feedback — it requires all 8 joints perfect simultaneously and is too strict "
        f"for participants with limited mobility. Judge each joint individually."
    )


@tool
def get_user_persona() -> str:
    """Return the stored user persona (name, age, gender, medical condition, notes).

    Call this after get_latest_session_results so you can frame feedback
    appropriately for the user's medical situation.
    If persona fields are null, ask the user for them early in conversation.
    """
    p = mem.get_persona()
    if all(v is None for v in p.values()):
        return "No persona stored yet. Ask the user for their name and medical context."
    return json.dumps(p, indent=2)


@tool
def update_user_persona(updates_json: str) -> str:
    """Save or update the user's persona in long-term memory.

    Call this when the user tells you their name, age, gender, or medical condition.
    Args:
        updates_json: JSON string with any subset of keys:
                      name, age (integer), gender, condition, notes.
                      Example: {"name": "Don", "age": 45, "condition": "frozen left shoulder"}
    """
    try:
        updates = json.loads(updates_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    result = mem.update_persona(**{k: v for k, v in updates.items() if v not in (None, "")})
    return f"Persona saved: {json.dumps(result, indent=2)}"


@tool
def get_session_history(n: int = 5) -> str:
    """Return the last *n* session cases from long-term memory for trend comparison.

    Call this after get_user_persona to see previous performance and the feedback
    you gave the user last time. Use it to identify trends and make comparisons.
    Args:
        n: number of previous sessions to retrieve (default 5)
    """
    sessions = mem.get_sessions(n)
    if not sessions:
        return "No previous sessions in memory. This appears to be the first session."

    lines = []
    for s in sessions:
        lines.append(
            f"--- {s['date']} ({s['day_of_week']}) — {s['exercise_name']} ---"
        )
        for jn, st in s.get("joint_stats", {}).items():
            sign = "+" if st["mean_diff"] >= 0 else ""
            lines.append(
                f"  {jn:<18}  bad={st['bad_pct']}%  "
                f"mean_diff={sign}{st['mean_diff']}°  "
                f"max_diff={st['max_abs_diff']}°"
            )
        lines.append(f"  FEEDBACK GIVEN: {s.get('feedback_given', 'none')}")
        lines.append("")

    return "\n".join(lines)


@tool
def save_session_to_memory(session_file: str, feedback: str) -> str:
    """Save the completed session and your planned feedback into long-term memory.

    MUST be called BEFORE giving the user their spoken feedback.
    Also automatically writes a trend comparison to the trend log if a previous
    session exists — you do not need a separate tool call for that.
    Args:
        session_file: filename from get_latest_session_results (e.g. session_20260511_143210_abc.json)
        feedback:     the feedback text you are about to speak (plain text, no markdown)
    """
    root = Path(__file__).parent
    path = root / session_file
    if not path.exists():
        matches = list(root.glob(f"**/{session_file}"))
        if not matches:
            return f"ERROR: Could not find session file '{session_file}'"
        path = matches[0]

    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        return f"ERROR reading session file: {exc}"

    meta  = data.get("meta", {})
    jstat = data.get("joint_stats", {})

    clean_stats = {
        jn: {
            "bad_pct":      s["bad_pct"],
            "mean_diff":    s["mean_diff"],
            "max_abs_diff": s["max_abs_diff"],
        }
        for jn, s in jstat.items()
    }

    # Snapshot previous sessions BEFORE adding this one
    previous_sessions = mem.get_sessions(1)
    previous = previous_sessions[-1] if previous_sessions else None

    case = mem.add_session(
        session_file    = session_file,
        joint_stats     = clean_stats,
        total_frames    = meta.get("total_frames") or meta.get("total_patient_frames", 0),
        angle_threshold = meta.get("angle_threshold", 25),
        feedback_given  = feedback,
        exercise_name   = EXERCISE_NAME,
    )

    # Auto-compute and log trend if a previous session exists
    trend_msg = "No previous session to compare."
    if previous:
        changes = []
        for jn, cur in clean_stats.items():
            prev_stat = previous["joint_stats"].get(jn)
            if prev_stat:
                delta = round(cur["bad_pct"] - prev_stat["bad_pct"], 1)
                direction = "improved" if delta < 0 else "worsened"
                changes.append(
                    f"{jn}: {direction} {abs(delta)}pp "
                    f"({prev_stat['bad_pct']}% → {cur['bad_pct']}%)"
                )
        trend_entry = {
            "current_date":     case["date"],
            "current_day":      case["day_of_week"],
            "compared_to_date": previous["date"],
            "compared_to_day":  previous["day_of_week"],
            "joint_changes":    changes,
            "agent_feedback":   feedback,
        }
        mem.append_trend_entry(trend_entry)
        trend_msg = (
            f"Trend logged: compared {case['date']} ({case['day_of_week']}) "
            f"vs {previous['date']} ({previous['day_of_week']}). "
            f"{len(changes)} joints compared."
        )

    return (
        f"Saved as case '{case['id']}' on {case['date']} ({case['day_of_week']}). "
        f"{trend_msg} Now speak your feedback to the user."
    )


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — built dynamically so persona is always current
# ─────────────────────────────────────────────────────────────────────────────

def _persona_block() -> str:
    """Render the USER PROFILE section from the current persona in memory."""
    p = mem.get_persona()
    name      = p.get("name")      or "unknown"
    age       = p.get("age")
    gender    = p.get("gender")    or "not specified"
    condition = p.get("condition") or "none recorded"
    notes     = p.get("notes")     or "none"
    age_str   = f"{age} years old" if age else "age not recorded"

    return (
        f"━━━ USER PROFILE ━━━\n"
        f"Name      : {name}\n"
        f"Age       : {age_str}\n"
        f"Gender    : {gender}\n"
        f"Condition : {condition}\n"
        f"Notes     : {notes}"
    )


def _build_prompt() -> str:
    return f"""You are a warm, encouraging exercise coach for young cancer survivors.
You communicate entirely by voice — plain spoken English only, no markdown, no asterisks,
no bullet points, no numbered lists, no bold, no headers.
Write exactly as you would speak to someone face to face.
Keep everyday replies to 2-4 sentences. Session feedback: 5-8 sentences maximum.

{_persona_block()}

━━━ CURRENT EXERCISE ━━━
Name: {EXERCISE_NAME}
Description: {EXERCISE_DESCRIPTION}

━━━ HOW THE METRICS WORK ━━━
Each joint angle is measured using three body landmarks at a keypoint vertex.
For example: right elbow angle = vectors from right_shoulder→right_elbow and right_elbow→right_wrist.

  diff = instructor_angle − patient_angle  (degrees, signed)

  bad_pct    — % of sampled frames where |diff| ≥ threshold (default 25°).
               PRIMARY quality indicator. High bad_pct = frequently off form on that joint.

  mean_diff  — average signed error.
               Negative (−): patient held joint HIGHER than instructor's angle.
               Positive (+): patient held joint LOWER than instructor's angle.
               WARNING: mean_diff near zero does NOT mean good form if bad_pct is high —
               the patient may be oscillating equally above and below (errors cancel out).

  max_abs_diff — worst single-frame deviation in the session.
               Values > 150° are almost certainly YOLO pose-detection glitches (one frame
               where the model lost the skeleton), NOT real movement errors. Never penalise these.

  NEVER use overall form accuracy (ok_pct) in feedback.
  It requires all 8 joints perfect simultaneously — far too strict for anyone with limited mobility.
  Always assess joints individually.

━━━ POST-SESSION WORKFLOW — exact order, no skipping ━━━
When the user says they have finished exercising:
  1. Call get_latest_session_results   → today's joint table + session filename.
  2. Call get_session_history(5)       → previous sessions for trend comparison.
  3. Compose your feedback (do NOT speak it yet — it must be saved first).
  4. Call save_session_to_memory       → pass the session filename + your composed feedback.
     This MUST happen before you speak. It also writes the trend log automatically.
  5. Speak your feedback to the user.

If save_session_to_memory returns an ERROR, tell the user — do not silently skip.
get_user_persona is available if you need to refresh profile details mid-conversation.

━━━ FEEDBACK RULES ━━━
  • Address the user by name in the opening sentence.
  • Read the Condition and Notes fields carefully before writing any feedback.
    — Any joint that overlaps with a recorded condition or injury should be treated with extra
      empathy. High bad_pct on an affected joint is expected — frame it as a physical reality,
      not a failure, and celebrate any improvement however small.
    — For conditions that limit range of motion, even moving through the exercise at all is
      an achievement worth naming explicitly.
    — Never suggest increasing intensity or effort for a condition that a care team should manage.
  • Name the 1-2 joints with the lowest bad_pct as clear strengths.
  • Name the 1-2 joints with the highest bad_pct as focus areas, framed constructively
    and always filtered through the user's known conditions.
  • Ignore any max_abs_diff > 150° — these are sensor noise, not real errors.
  • Trend comparison: if previous sessions exist, state one concrete improvement and one
    area still developing, using actual numbers from both sessions.
    Example: "Your right shoulder went from 35% last Thursday down to 18% today."
  • Close with a genuine one-sentence encouragement.
  • Never invent numbers — only quote figures from the tool results.
  • Compare the user only to their own previous sessions, never to a healthy baseline.

━━━ PERSONA UPDATES ━━━
  If the user shares new medical information during conversation, call update_user_persona
  immediately so it is remembered for next time."""


# ─────────────────────────────────────────────────────────────────────────────
# Agent + REPL
# ─────────────────────────────────────────────────────────────────────────────

def build_agent():
    model = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0,
    )
    return create_agent(
        model=model,
        tools=[
            launch_exercise_app,
            get_latest_session_results,
            get_user_persona,
            update_user_persona,
            get_session_history,
            save_session_to_memory,  # also writes trend log automatically
        ],
        system_prompt=_build_prompt(),
        checkpointer=MemorySaver(),
    )


def run_agent():
    agent  = build_agent()
    config = {"configurable": {"thread_id": "exercise-session"}}

    print("\n" + "=" * 60)
    print("  Smart Mirror — Exercise Coaching Agent")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in ("quit", "exit", ""):
            print("Goodbye!")
            break
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print(f"\nCoach: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    run_agent()
