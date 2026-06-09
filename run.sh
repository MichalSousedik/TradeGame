#!/bin/bash
# TradeGame — start live loop + dashboard in one command
#
#   ./run.sh              BTC/USDT 1h (defaults from config.yaml)
#   ./run.sh --timeframe 1m           fast test mode (bar every minute)
#   ./run.sh --symbol ETH/USDT

PYTHON=/Library/Developer/CommandLineTools/usr/bin/python3
STREAMLIT=/Users/michalsousedik/Library/Python/3.9/bin/streamlit
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== TradeGame ==="
echo "Starting live loop in background..."
$PYTHON -m tradegame.live.runner "$@" &
LOOP_PID=$!
echo "Loop PID: $LOOP_PID"

echo "Starting dashboard at http://localhost:8501"
echo "Press Ctrl-C to stop both."
trap "echo 'Stopping...'; kill $LOOP_PID 2>/dev/null; exit 0" INT TERM

$STREAMLIT run "$DIR/src/tradegame/dashboard/app.py" --server.headless true

kill $LOOP_PID 2>/dev/null
