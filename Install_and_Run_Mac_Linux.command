#!/bin/bash
echo "==================================================="
echo "WhisperX Pro - Mac/Linux Installer and Launcher"
echo "==================================================="

# Move into the folder where the script is located
cd "$(dirname "$0")"

# Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Checking for updates and installing dependencies..."
echo "[INFO] Note: The first run might take 5-10 minutes to download AI models (like PyTorch and Whisper)."
python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo "[INFO] Launching WhisperX Pro..."
streamlit run whisper_app.py
