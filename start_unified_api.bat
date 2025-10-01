@echo off
echo Starting Unified Translation, SEO, and Tagging API...
echo.

REM Check if .env file exists
if not exist .env (
    echo ⚠️  .env file not found!
    echo Please copy env_example.txt to .env and add your OpenAI API key
    echo.
    echo Example:
    echo     copy env_example.txt .env
    echo     edit .env
    echo.
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist venv (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

echo 📦 Installing dependencies...
pip install -r requirements.txt

echo 🚀 Starting API server...
echo.
echo The API will be available at:
echo   - Health: http://localhost:5000/health
echo   - Test:   http://localhost:5000/test
echo   - Process: http://localhost:5000/process
echo.
echo Press Ctrl+C to stop the server
echo.

python unified_api.py

pause

