#!/usr/bin/env bash
set -x
cd "$(dirname "$0")/../vendor/MCP-Universe"
uv venv -q --python 3.12 .venv --seed
export VIRTUAL_ENV=$PWD/.venv PATH=$PWD/.venv/bin:$PATH UV_HTTP_TIMEOUT=180
uv pip install -q -r requirements.txt
uv pip install -q -e . mcp-server-fetch mcp-server-calculator
