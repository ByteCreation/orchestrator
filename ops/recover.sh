#!/usr/bin/env bash
# orchestrator-recover.sh — manual recovery if the orchestrator won't start.
#
# Use when imggen-run's restore fails or the orchestrator is stuck.
#
# Usage:
#   /opt/orchestrator/ops/recover.sh

set -uo pipefail

echo "[orchestrator-recover] Killing any stuck orchestrator processes..."
sudo systemctl stop orchestrator.service 2>/dev/null || true
sleep 2

# Free port 8010 if something is holding it
if command -v fuser >/dev/null 2>&1; then
  sudo fuser -k 8010/tcp 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  lsof -ti:8010 | xargs -r kill -9 2>/dev/null || true
fi
sleep 2

echo "[orchestrator-recover] Starting orchestrator..."
sudo systemctl start orchestrator.service
sleep 5

echo "[orchestrator-recover] Verifying..."
for i in $(seq 1 10); do
  if curl -sf http://127.0.0.1:8010/health >/dev/null 2>&1; then
    echo "[orchestrator-recover] OK — orchestrator running on :8010"
    exit 0
  fi
  echo "[orchestrator-recover] waiting... attempt $i/10"
  sleep 3
done

echo "[orchestrator-recover] FAILED — orchestrator did not come up." >&2
echo "[orchestrator-recover] Check: sudo journalctl -u orchestrator.service -n 50" >&2
exit 1