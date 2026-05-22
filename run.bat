@echo off
echo Starting FastAPI RAG Server...
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
pause
