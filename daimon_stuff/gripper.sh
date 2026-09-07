#!/usr/bin/env bash
set -euo pipefail

command="${1:-}"
if [[ "$command" != "grip" && "$command" != "release" && "$command" != "status" && "$command" != "serve" ]]; then
  echo "Usage: $0 {grip|release|status|serve} [extra args...]" >&2
  exit 2
fi
shift || true

cd "$(dirname "$0")/.."
exec python daimon_stuff/grip_signal_receiver.py --command "$command" "$@"
