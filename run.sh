#!/bin/bash
# ═══════════════════════════════════════════════════════
#  Employee Monitor — Start Everything
#  Usage:  ./run.sh
# ═══════════════════════════════════════════════════════

# set -e removed — pkill returns 1 when nothing to kill, which caused early exit
cd "$(dirname "$0")"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       Employee Monitor — Starting Up          ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check Python ──────────────────────────────
echo -e "${CYAN}[1/3] Checking Python...${NC}"
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}  ERROR: python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
PY=$(python3 --version 2>&1)
echo -e "${GREEN}  ✓ $PY${NC}"

# ── Step 2: Install dependencies ─────────────────────
echo ""
echo -e "${CYAN}[2/3] Installing dependencies...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}  ✓ All packages ready${NC}"

# ── Kill ALL previous instances (force, all duplicates) ──
echo ""
echo -e "${CYAN}  Stopping any previous instances...${NC}"
pkill -9 -f "monitor.py"  2>/dev/null || true; sleep 1
pkill -9 -f "web_ui.py"   2>/dev/null || true; sleep 1
pkill -9 -f "gunicorn"    2>/dev/null || true; sleep 1
fuser -k 5000/tcp         2>/dev/null || true
sleep 1
echo -e "${GREEN}  ✓ Clean start${NC}"

# ── Step 3: Get local IP for browser URL ─────────────
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$LOCAL_IP" ] && LOCAL_IP="localhost"

# ── Cleanup on exit ───────────────────────────────────
MONITOR_PID=""
UI_PID=""

cleanup() {
    echo ""
    echo -e "${YELLOW}  Stopping all services...${NC}"
    [ -n "$MONITOR_PID" ] && kill "$MONITOR_PID" 2>/dev/null && echo -e "${GREEN}  ✓ Monitor stopped${NC}"
    [ -n "$UI_PID" ]      && kill "$UI_PID"      2>/dev/null && echo -e "${GREEN}  ✓ Web UI stopped${NC}"
    echo -e "${BOLD}  Done. Goodbye.${NC}"
    echo ""
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Start monitor.py ──────────────────────────────────
echo ""
echo -e "${CYAN}[3/3] Starting services...${NC}"
echo ""

python3 monitor.py > logs/monitor.log 2>&1 &
MONITOR_PID=$!
sleep 2

if kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo -e "${GREEN}  ✓ AI Monitor running   (PID $MONITOR_PID)${NC}"
    echo -e "    Log: tail -f logs/monitor.log"
else
    echo -e "${RED}  ✗ Monitor failed to start — check logs/monitor.log${NC}"
fi

# ── Start web_ui.py ───────────────────────────────────
# Use gunicorn for production-grade speed (handles 30 threads vs Flask's 1)
if command -v gunicorn &>/dev/null; then
    gunicorn --workers=1 --threads=50 --worker-class=gthread \
             --bind=0.0.0.0:5000 --timeout=30 --keep-alive=2 \
             --log-file=logs/web_ui.log --log-level=warning \
             web_ui:app > /dev/null 2>&1 &
else
    python3 web_ui.py > logs/web_ui.log 2>&1 &
fi
UI_PID=$!
sleep 3

if kill -0 "$UI_PID" 2>/dev/null; then
    echo -e "${GREEN}  ✓ Web Dashboard running (PID $UI_PID)${NC}"
    echo -e "    Log: tail -f logs/web_ui.log"
else
    echo -e "${RED}  ✗ Web UI failed to start — check logs/web_ui.log${NC}"
fi

# ── Show access info ──────────────────────────────────
echo ""
# Show model status
if [ -f "custom_model/weights/best.pt" ]; then
    echo -e "${GREEN}  ✓ Custom trained model found — using it!${NC}"
else
    echo -e "${YELLOW}  ℹ Using base yolov8s model. Go to /annotate to train.${NC}"
fi

echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Open your browser and go to:                ║${NC}"
echo -e "${BOLD}║                                               ║${NC}"
echo -e "${BOLD}║   ${CYAN}http://$LOCAL_IP:5000${BOLD}           Dashboard ║${NC}"
echo -e "${BOLD}║   ${CYAN}http://$LOCAL_IP:5000/annotate${BOLD}  Training  ║${NC}"
echo -e "${BOLD}║                                               ║${NC}"
echo -e "${BOLD}║   Press  Ctrl+C  to stop everything           ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# ── Tail both logs together ───────────────────────────
echo -e "${YELLOW}  Live output (Ctrl+C to stop):${NC}"
echo -e "  ─────────────────────────────"
tail -f logs/monitor.log logs/web_ui.log 2>/dev/null &
TAIL_PID=$!

# ── Wait forever ──────────────────────────────────────
wait "$MONITOR_PID" "$UI_PID" 2>/dev/null
kill "$TAIL_PID" 2>/dev/null
cleanup
