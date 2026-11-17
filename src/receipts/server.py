"""Web server: streams the research trace live, then the final result, as NDJSON.

POST /api/run     {"mode": "research"|"audit", "input": "..."}  -> {"event": ...} lines, then {"result": ...}
GET  /api/replay  replays the newest saved run (or the bundled sample) with its original pacing
GET  /api/health  which keys are configured (never the keys themselves)
"""

from __future__ import annotations

import datetime as dt
import json
import queue
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import build
from .render import to_markdown

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
RUNS = ROOT / "runs"
SAMPLE = ROOT / "tests" / "fixtures" / "sample_run.json"

app = FastAPI(title="Receipts")


class RunRequest(BaseModel):
    mode: str = Field("research", pattern="^(research|audit)$")
    input: str = Field(..., min_length=3, max_length=12000)


def _line(obj: dict) -> bytes:
    return (json.dumps(obj, default=str) + "\n").encode("utf-8")


@app.get("/api/health")
def health():
    s = Settings.from_env()
    missing = s.missing_keys()
    return {
        "ready": not missing,
        "missing": missing,
        "models": {"fast": s.model_fast, "smart": s.model_smart, "fallback": s.model_fallback},
        "replay_available": SAMPLE.exists() or any(RUNS.glob("*.json")),
    }


@app.post("/api/run")
def run(req: RunRequest):
    settings = Settings.from_env()
    missing = settings.missing_keys()
    if missing:
        return JSONResponse(
            status_code=400,
            content={"error": f"Add {' and '.join(missing)} to .env, then restart the server."},
        )

    events: queue.Queue = queue.Queue()
    done = object()

    def work():
        try:
            agent = build(settings, listener=lambda e: events.put({"event": e}))
            result = agent.audit(req.input) if req.mode == "audit" else agent.run(req.input)
            _save(result)
            events.put({"result": result.to_dict()})
        except Exception as e:  # noqa: BLE001 — report, never hang the stream
            events.put({"error": f"{type(e).__name__}: {str(e)[:300]}"})
        finally:
            events.put(done)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is done:
                return
            yield _line(item)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/replay")
def replay(speed: float = 4.0):
    """Re-stream a saved run so the UI can be shown without spending credits."""
    saved = sorted(RUNS.glob("*.json"))
    path = saved[-1] if saved else SAMPLE
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "No saved runs yet. Run a question first."})
    data = json.loads(path.read_text(encoding="utf-8"))

    def stream():
        yield _line({"replay": path.name})
        last = 0.0
        for e in data.get("trace", []):
            time.sleep(max(0.0, min(1.5, (e.get("t", last) - last) / max(speed, 0.1))))
            last = e.get("t", last)
            yield _line({"event": e})
        yield _line({"result": data})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _save(result) -> None:
    RUNS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (RUNS / f"{stamp}.json").write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    (RUNS / f"{stamp}.md").write_text(to_markdown(result), encoding="utf-8")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
