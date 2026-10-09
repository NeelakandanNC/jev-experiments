#!/usr/bin/env bash
# Icon sets, CSS frameworks and fonts used by the synthetic UI generator (bench/synth.py). npm only.
set -euo pipefail
cd "$(dirname "$0")/.."
A=data/assets; mkdir -p "$A"
fetch() {  # npm package@version -> data/assets/<dir>
  [ -d "$A/$2" ] && return
  local tgz; tgz=$(cd "$A" && npm pack -q "$1" 2>/dev/null | tail -1)
  mkdir -p "$A/$2" && tar -xzf "$A/$tgz" -C "$A/$2" --strip-components=1 && rm "$A/$tgz"
}
fetch bootstrap-icons@1.11.3 bootstrap-icons
fetch lucide-static@0.460.0 lucide
fetch @mdi/svg@7.4.47 mdi
fetch bootstrap@5.3.3 bootstrap
fetch bulma@1.0.2 bulma
fetch @picocss/pico@2.0.6 pico
fetch @fontsource/roboto@5.1.0 font-roboto
fetch @fontsource/open-sans@5.1.0 font-open-sans
fetch @fontsource/lato@5.1.0 font-lato
echo "assets in $A"
