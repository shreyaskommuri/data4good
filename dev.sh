#!/bin/bash
# Launch backend (FastAPI) and frontend (React/Vite) together.
# Usage: ./dev.sh

cd "$(dirname "$0")"

cleanup() {
  kill $BACK_PID $FRONT_PID 2>/dev/null
  exit
}
trap cleanup INT TERM

# Backend
source venv/bin/activate 2>/dev/null
pip install -q fastapi uvicorn 2>/dev/null
uvicorn api:app --reload --port 8000 &
BACK_PID=$!

# Frontend
cd frontend
npm install --silent 2>/dev/null
npm run dev &
FRONT_PID=$!

echo ""
echo "=================================="
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "  Press Ctrl+C to stop both"
echo "=================================="
echo ""

wait
