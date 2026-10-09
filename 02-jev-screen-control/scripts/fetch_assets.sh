#!/usr/bin/env bash
# Icon sets, CSS frameworks and fonts used by the synthetic UI generator (bench/synth.py).
# Straight from the npm registry with curl (no node needed).
set -euo pipefail
cd "$(dirname "$0")/.."
A=data/assets; mkdir -p "$A"
fetch() {  # <tarball path on registry.npmjs.org> <dir>
  [ -d "$A/$2" ] && return
  mkdir -p "$A/$2"
  curl -fsSL "https://registry.npmjs.org/$1" | tar -xz -C "$A/$2" --strip-components=1
}
fetch bootstrap-icons/-/bootstrap-icons-1.11.3.tgz bootstrap-icons
fetch lucide-static/-/lucide-static-0.460.0.tgz lucide
fetch @mdi/svg/-/svg-7.4.47.tgz mdi
fetch bootstrap/-/bootstrap-5.3.3.tgz bootstrap
fetch bulma/-/bulma-1.0.2.tgz bulma
fetch @picocss/pico/-/pico-2.0.6.tgz pico
fetch @fontsource/roboto/-/roboto-5.1.0.tgz font-roboto
fetch @fontsource/open-sans/-/open-sans-5.1.0.tgz font-open-sans
fetch @fontsource/lato/-/lato-5.1.0.tgz font-lato
echo "assets in $A"
