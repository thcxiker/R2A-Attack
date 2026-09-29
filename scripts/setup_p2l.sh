#!/usr/bin/env bash
# P2L's model code (https://github.com/lmarena/p2l) is not redistributed here:
# fetch it into external/p2l (pinned to the commit R2A was tested with).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d external/p2l ]]; then
  git clone https://github.com/lmarena/p2l.git external/p2l
fi
git -C external/p2l checkout --quiet a905fa5
echo "P2L code ready in external/p2l (or set R2A_P2L_PATH to another checkout)."
