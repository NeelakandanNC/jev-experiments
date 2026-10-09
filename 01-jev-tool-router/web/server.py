"""Minimal local dashboard: add keys, run each benchmark, see the results.

    .venv/bin/python -m web.server        -> http://127.0.0.1:8790

Each Run is a routing eval (bench.routing) of the 4 variations: Jev and OpenAI's decision model,
both through the Decisions API, each handing the agent its top 5 or top 10 tools. A run uses
whichever models have a key; each model's latest results are shown side by side.
Baselines (BM25, embeddings, LLM) and other settings: use the CLI.
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
from jevroute.decide import has_key  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
ENV_FILE = ROOT / ".env"
RESULTS = ROOT / "results" / "routing"

# ks: shortlist sizes evaluated and shown (k=1 is always run too, for the calibration numbers).
# MCP-Bench tasks need ~9 tools each, so a 5-tool shortlist can't cover them: top-10 only there.
BENCHMARKS = {
    "atlas": {"name": "MCP-Atlas", "by": "Scale AI", "ks": [5, 10]},
    "mcpbench": {"name": "MCP-Bench", "by": "Accenture", "ks": [10]},
    "universe": {"name": "MCP-Universe", "by": "Salesforce", "ks": [5, 10]},
}
OPENAI_DECISIONS = os.getenv("JEVROUTE_OPENAI_DECISION_MODEL", "openai/gpt-6-luna")
MODELS = {"jev": "typesafe-ai/jev", f"decision:{OPENAI_DECISIONS}": OPENAI_DECISIONS}  # router spec -> model
KEYS = {"openai": "OPENAI_API_KEY", "gateway": "AI_GATEWAY_API_KEY"}

app = FastAPI(title="jevroute")
procs: dict[str, tuple[subprocess.Popen, str, str]] = {}

for k, v in load_env().items():
    os.environ.setdefault(k, v)


def _merged(bench: str) -> dict | None:
    """Latest finished result per model across runs, so models run at different times line up."""
    merged: dict = {}
    for p in sorted(RESULTS.glob(f"{bench}-*/summary.json"), key=lambda p: p.stat().st_mtime):
        s = json.loads(p.read_text())
        for name, v in s.get("routers", {}).items():
            if v.get("cases") and v.get("errors", 0) < v["cases"]:
                merged.setdefault("routers", {})[name] = {**v, "run_id": s["run_id"]}
                merged.update({k: s[k] for k in ("ks", "catalog_tokens", "cases", "tasks", "catalog_tools")})
    return merged or None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def state():
    out = {}
    for b, meta in BENCHMARKS.items():
        info: dict = dict(meta)
        cp = catalog_path(b)
        if cp.exists():
            c = Catalog.load(cp)
            info["catalog"] = {"tools": len(c), "servers": len(c.servers), "tokens": c.tokens()}
        p = procs.get(b)
        info["running"] = running = p is not None and p[0].poll() is None
        if running:
            prog = RESULTS / p[2] / "progress.json"
            info["progress"] = json.loads(prog.read_text()) if prog.exists() else {"done": 0, "total": 0}
        elif p and p[0].returncode:
            info["error"] = Path(p[1]).read_text(errors="replace")[-600:]
        info["result"] = _merged(b)
        out[b] = info
    return {"keys": {k: bool(os.environ.get(v)) for k, v in KEYS.items()},
            "models_ready": [spec for spec, model in MODELS.items() if has_key(model)],
            "mock": os.environ.get("JEVROUTE_MOCK", "") in ("1", "true"), "benchmarks": out}


class KeyIn(BaseModel):
    which: str  # openai | gateway
    key: str


@app.post("/api/key")
def set_key(body: KeyIn):
    var = KEYS.get(body.which)
    key = body.key.strip()
    if not var:
        raise HTTPException(400, "unknown key")
    if len(key) < 10 or any(c.isspace() for c in key):
        raise HTTPException(400, "that does not look like an API key")
    lines = [l for l in (ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []) if not l.startswith(f"{var}=")]
    ENV_FILE.write_text("\n".join(lines + [f"{var}={key}"]) + "\n")
    ENV_FILE.chmod(0o600)
    os.environ[var] = key
    return {"ok": True}


@app.post("/api/run/{bench}")
def run(bench: str):
    if bench not in BENCHMARKS:
        raise HTTPException(404)
    if not catalog_path(bench).exists():
        raise HTTPException(400, "tool catalog not built yet")
    if bench in procs and procs[bench][0].poll() is None:
        raise HTTPException(409, "already running")
    ready = [spec for spec, model in MODELS.items() if has_key(model) or os.environ.get("JEVROUTE_MOCK")]
    if not ready:
        raise HTTPException(400, "add a key first")
    done = set(((_merged(bench) or {}).get("routers") or {}))
    # models that have no result yet; once every ready model has one, Run re-runs them all
    routers = [spec for spec in ready if MODELS[spec].split("/")[-1] not in done] or ready
    run_id = time.strftime(f"{bench}-%Y%m%d-%H%M%S")
    logs = ROOT / "results" / "jobs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{run_id}.log"
    cmd = [sys.executable, "-m", "bench.routing", "--benchmark", bench, "--routers", ",".join(routers),
           "--ks", ",".join(map(str, [1, *BENCHMARKS[bench]["ks"]])), "--run-id", run_id]
    p = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                         stdout=log.open("w"), stderr=subprocess.STDOUT)
    procs[bench] = (p, str(log), run_id)
    return {"run_id": run_id, "routers": routers}


def main():
    import uvicorn
    port = int(os.getenv("JEVROUTE_DASHBOARD_PORT", 8790))
    print(f"jevroute dashboard on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
