@echo off
echo ============================================
echo   ROOTECH RAG Server (Python FastAPI)
echo ============================================

cd /d "%~dp0"

REM === Set your conda python path here ===
set PYTHON_CMD=C:\Users\ADMIN\miniconda3\envs\VisionTest\python.exe

echo Python: %PYTHON_CMD%
echo.

%PYTHON_CMD% -c "import fastapi; import uvicorn" 2>nul
if errorlevel 1 (
    echo [INFO] Installing fastapi and uvicorn...
    %PYTHON_CMD% -m pip install fastapi uvicorn --quiet
)

echo Starting RAG Server...
echo (Model loading takes 30s - 1min)
echo.

%PYTHON_CMD% scripts/rag_server.py --data_dir data --port 5000 --device cpu --llm_model cyankiwi/Qwen3.5-4B-AWQ-4bit

pause
