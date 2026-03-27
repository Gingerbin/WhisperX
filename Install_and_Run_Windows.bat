@echo off
echo ===================================================
echo WhisperX Pro - Windows Installer and Launcher
echo ===================================================

echo [INFO] If you are running this inside Dropbox, OneDrive, or Google Drive,
echo [INFO] installation may fail due to file syncing locks. If errors occur,
echo [INFO] move this folder to your Desktop or C:\ drive and try again.
echo.

:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org and ensure you check "Add Python to PATH" during installation.
    pause
    exit /b
)

:: Check if virtual environment exists
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing dependencies...
echo [INFO] Note: The first run might take 5-10 minutes to download AI models (like PyTorch and Whisper).

:: Suppress pip upgrade errors if files are locked by Windows Defender/Dropbox
python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r requirements.txt

echo [INFO] Launching WhisperX Pro...
python -m streamlit run whisper_app.py

pause
