"""
main.py — entry point for the Smart Mirror exercise coaching system.

Usage:
    uv run python main.py            # start voice chat UI  (default)
    uv run python main.py chat       # same as above
    uv run python main.py server     # start exercise tracking server only
    uv run python main.py agent      # start text-only REPL agent (no UI)
"""

import sys


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if mode in ("chat", "ui"):
        import threading, webbrowser, uvicorn
        print("Starting Smart Mirror chat UI on http://localhost:8002")
        threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8002")).start()
        uvicorn.run("chat_server:app", host="0.0.0.0", port=8002, reload=False)

    elif mode == "server":
        import uvicorn
        print("Starting Smart Mirror exercise server on http://localhost:8001")
        uvicorn.run("realtime_server:app", host="0.0.0.0", port=8001, reload=False)

    elif mode == "agent":
        from agent import run_agent
        run_agent()

    else:
        print(f"Unknown mode '{mode}'. Use: chat | server | agent")
        sys.exit(1)


if __name__ == "__main__":
    main()
