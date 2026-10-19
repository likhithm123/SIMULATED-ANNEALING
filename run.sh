#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PORT="${PORT:-8008}"

echo "=========================================================="
echo "   ADAPTIVE SIMULATED ANNEALING OPTIMIZATION PLATFORM"
echo "=========================================================="

if [ -f "$DIR/backend/venv/bin/activate" ]; then
    echo "Activating virtual environment at $DIR/backend/venv..."
    source "$DIR/backend/venv/bin/activate"
fi

echo "Checking dependencies..."
python -m pip install -q -r "$DIR/backend/requirements.txt"

echo "Starting server on port $PORT..."
echo ""
echo "👉 Open your browser at: http://127.0.0.1:$PORT"
echo "   (Press Ctrl+C to stop the server)"
echo "=========================================================="

cd "$DIR/backend"
python -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
