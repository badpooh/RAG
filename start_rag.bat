@echo off
echo ============================================
echo   ROOTECH RAG Server (Python FastAPI)
echo ============================================

cd /d "%~dp0"

REM === Set your conda python path here / company ===
REM set PYTHON_CMD=C:\Users\ADMIN\miniconda3\envs\VisionTest\python.exe

REM === Set your conda python path here / home ===
set PYTHON_CMD=C:\Users\JOYS\miniconda3\envs\vision\python.exe

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

REM === VLLM ===
REM %PYTHON_CMD% scripts/rag_server_ollama.py --data_dir data --port 5000 --device cpu --llm_model cyankiwi/Qwen3.5-4B-AWQ-4bit

REM === Ollama ===
%PYTHON_CMD% scripts/rag_server_ollama.py --data_dir data --port 5000 --device cpu --llm_model qwen3.5:4b

pause
