@echo off
echo ============================================
echo   ROOTECH Web Server (Java)
echo ============================================

cd /d "%~dp0"

echo [1/2] Compiling...
if not exist build mkdir build
javac -encoding UTF-8 src/RootechServer.java -d build
if errorlevel 1 (
    echo Compilation failed!
    pause
    exit /b 1
)

echo [2/2] Starting web server...
cd build
java RootechServer
pause
