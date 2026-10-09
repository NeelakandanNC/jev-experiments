"""Minimal local dashboard: paste the AI Gateway key, run each benchmark, see the results.

    .venv/bin/python -m web.server        -> http://127.0.0.1:8790

Each Run is a routing eval (bench.routing) of the 4 variations: Jev and OpenAI's decision model,
both through the OpenAI-compatible Decisions API on AI Gateway, each handing the agent its
top 5 or top 10 tools. Baselines (BM25, embeddings, LLM) and other settings: use the CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.datasets import catalog_path  # noqa: E402
from bench.e2e import load_env  # noqa: E402
from jevroute.catalog import Catalog  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
ENV_FILE = ROOT / ".env"
RESULTS = ROOT / "results" / "routing"

BENCHMARKS = {
    "atlas": {"name": "MCP-Atlas", "by": "Scale AI", "args": ["--limit-tasks", "100"]},
    "mcpbench": {"name": "MCP-Bench", "by": "Accenture", "args": []},
    "universe": {"name": "MCP-Universe", "by": "Salesforce", "args": []},
}
# The 4 variations: two decision models x top-5 / top-10 (k=1 is kept for the calibration numbers).
OPENAI_DECISIONS = os.getenv("JEVROUTE_OPENAI_DECISION_MODEL", "openai/gpt-6-luna-decisions")
ROUTERS = f"jev,decision:{OPENAI_DECISIONS}"
KS = "1,5,10"

app = FastAPI(title="jevroute")
procs: dict[str, tuple[subprocess.Popen, str]] = {}

for k, v in load_env().items():
    os.environ.setdefault(k, v)


def _latest(bench: str) -> tuple[Path | None, dict | None]:
    runs = sorted(RESULTS.glob(f"{bench}-*/config.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in runs:
        d = p.parent
        s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        return d, s
    return None, None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def state():
    out = {}
    for b, meta in BENCHMARKS.items():
        info: dict = {**{k: meta[k] for k in ("name", "by")}}
        cp = catalog_path(b)
        if cp.exists():
            c = Catalog.load(cp)
            info["catalog"] = {"tools": len(c), "servers": len(c.servers), "tokens": c.tokens()}
        p = procs.get(b)
        running = p is not None and p[0].poll() is None
        run_dir, summary = _latest(b)
        if running and run_dir:
            prog = run_dir / "progress.json"
            info["progress"] = json.loads(prog.read_text()) if prog.exists() else {"done": 0, "total": 0}
        if p and not running and p[0].returncode:
            info["error"] = Path(p[1]).read_text(errors="replace")[-600:]
        info["running"] = running
        if summary and not running:
            info["result"] = summary
        out[b] = info
    return {"key_set": bool(os.environ.get("AI_GATEWAY_API_KEY")),
            "mock": os.environ.get("JEVROUTE_MOCK", "") in ("1", "true"), "benchmarks": out}


class KeyIn(BaseModel):
    key: str


@app.post("/api/key")
def set_key(body: KeyIn):
    key = body.key.strip()
    if len(key) < 10 or any(c.isspace() for c in key):
        raise HTTPException(400, "that does not look like an API key")
    lines = [l for l in (ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else [])
             if not l.startswith("AI_GATEWAY_API_KEY=")]
    ENV_FILE.write_text("\n".join(lines + [f"AI_GATEWAY_API_KEY={key}"]) + "\n")
    ENV_FILE.chmod(0o600)
    os.environ["AI_GATEWAY_API_KEY"] = key
    return {"ok": True}


@app.post("/api/run/{bench}")
def run(bench: str):
    if bench not in BENCHMARKS:
        raise HTTPException(404)
    if not catalog_path(bench).exists():
        raise HTTPException(400, "tool catalog not built yet")
    if bench in procs and procs[bench][0].poll() is None:
        raise HTTPException(409, "already running")
    run_id = time.strftime(f"{bench}-%Y%m%d-%H%M%S")
    logs = ROOT / "results" / "jobs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{run_id}.log"
    cmd = [sys.executable, "-m", "bench.routing", "--benchmark", bench, "--routers", ROUTERS,
           "--ks", KS, "--run-id", run_id, *BENCHMARKS[bench]["args"]]
    p = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                         stdout=log.open("w"), stderr=subprocess.STDOUT)
    procs[bench] = (p, str(log))
    return {"run_id": run_id}


def main():
    import uvicorn
    port = int(os.getenv("JEVROUTE_DASHBOARD_PORT", 8790))
    print(f"jevroute dashboard on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
