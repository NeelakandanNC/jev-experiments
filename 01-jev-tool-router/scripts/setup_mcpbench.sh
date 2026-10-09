#!/usr/bin/env bash
# MCP-Bench in its own venv. Its requirements float to mcp 2.x, which its servers predate,
# so we pin the Aug-2025 stack (mcp 1.13.1 / pydantic 2.11.7 / fastmcp 2.11.3). scipy 1.14.1
# because newer macOS rejects the 1.15 wheel's binary.
set -x
cd "$(dirname "$0")/../vendor/mcp-bench"
uv venv -q --python 3.10 .venv --seed
export VIRTUAL_ENV=$PWD/.venv PATH=$PWD/.venv/bin:$PATH UV_HTTP_TIMEOUT=180
C=$PWD/.venv/constraints.txt
printf 'mcp==1.13.1\npydantic==2.11.7\nscipy==1.14.1\nfastmcp==2.11.3\n' > "$C"
pip install -q -r mcp_servers/requirements.txt -c "$C"
(cd mcp_servers && bash ./install.sh)
S=mcp_servers
for d in unit-converter-mcp paper-search-mcp time-mcp; do uv pip install -q -c "$C" -e $S/$d; done
uv pip install -q -c "$C" -r $S/mcp-osint-server/requirements.txt -e $S/mcp-osint-server beautifulsoup4 lxml openai pyyaml python-dotenv
uv pip uninstall -q fastmcp-slim mcp-types 2>/dev/null || true
uv pip install -q -c "$C" --reinstall "mcp[cli]==1.13.1" fastmcp==2.11.3 scipy==1.14.1
(cd $S/okx-mcp && npm i -q --no-audit --no-fund --save-dev @types/node && npm run build)
(cd $S/openapi-mcp-server && npm install --no-audit --no-fund)
(cd $S/mcp-google-map && npm install --no-audit --no-fund && npm run build)
(cd $S/biomcp && uv sync -q)
python ./utils/collect_mcp_info.py | tail -3   # expect: 28/28 servers connected, 257 tools
