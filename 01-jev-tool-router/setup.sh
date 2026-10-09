#!/usr/bin/env bash
# One-time setup: this experiment's venv + the three benchmark repos (pinned to the commits we tested).
set -euo pipefail
cd "$(dirname "$0")"
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
mkdir -p vendor
clone() { [ -d "vendor/$2" ] || git clone -q https://github.com/$1.git "vendor/$2"; git -C "vendor/$2" checkout -q "$3"; }
clone scaleapi/mcp-atlas mcp-atlas "$(cat pins/mcp-atlas)"
clone Accenture/mcp-bench mcp-bench "$(cat pins/mcp-bench)"
clone SalesforceAIResearch/MCP-Universe MCP-Universe "$(cat pins/MCP-Universe)"
echo "Routing eval is ready (catalogs ship in data/catalogs). For end-to-end runs see README > End-to-end."
