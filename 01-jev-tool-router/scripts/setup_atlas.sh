#!/usr/bin/env bash
# Python deps + TypeScript harness. The MCP servers themselves run in Scale's Docker sandbox:
#   docker run --rm -p 1984:1984 --env-file vendor/mcp-atlas/.env ghcr.io/scaleapi/mcp-atlas:1.2.7
set -x
cd "$(dirname "$0")/../vendor/mcp-atlas"
uv venv -q --python 3.12 .venv --seed
UV_HTTP_TIMEOUT=180 uv pip install -q --python .venv/bin/python -r requirements.txt
cd services/agent-harness && npm install --no-audit --no-fund
