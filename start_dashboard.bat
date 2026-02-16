@echo off
REM Quick Start Script for Credit Risk Dashboard
REM This script starts both the FastAPI backend and Streamlit dashboard

echo ========================================
echo Credit Risk Dashboard - Quick Start
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "src\api\main.py" (
    echo Error: Please run this script from the credit-risk-model directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo [1/2] Starting FastAPI Backend on port 8000...
echo.

REM Start FastAPI in a new window
start "Credit Risk API" cmd /k "python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a bit for the API to start
timeout /t 5 /nobreak > nul

echo [2/2] Starting Streamlit Dashboard on port 8501...
echo.

REM Start Streamlit in a new window
start "Credit Risk Dashboard" cmd /k "streamlit run dashboard\app.py"

echo.
echo ========================================
echo Both services are starting...
echo ========================================
echo.
echo API Backend:      http://localhost:8000
echo API Docs:         http://localhost:8000/docs
echo Streamlit Dashboard: http://localhost:8501
echo.
echo Press any key to open dashboard in browser...
pause > nul

REM Open dashboard in default browser
start http://localhost:8501

echo.
echo Dashboard opened in browser!
echo.
echo To stop the services:
echo - Close the terminal windows titled "Credit Risk API" and "Credit Risk Dashboard"
echo - Or press Ctrl+C in each terminal window
echo.
pause
