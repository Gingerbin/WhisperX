# WhisperX Pro (Cloud LLM Edition)

A standalone Streamlit web application that provides advanced transcription, diarization (speaker separation), acoustic volume analysis, VADER sentiment detection, psychological state tracking, and Cloud LLM Intent Analysis. 

It supports both **CPU-Optimized (whisperX)** and **GPU/Apple Silicon-Optimized (insanely-fast-whisper)** backend processing engines!

## 🚀 How to Install and Run

You do **not** need to open a terminal to use this application. Just use the included one-click installers!

### 💻 For Windows Users:
1. Ensure you have [Python 3.10+](https://www.python.org/downloads/) installed. (Make sure you check the **"Add Python to PATH"** box during installation).
2. Download or Clone this repository.
3. Double-click the **`Install_and_Run_Windows.bat`** file.
4. *Note: The first time you run this, it will automatically download PyTorch, the Whisper models, and create a virtual environment. This may take 5-10 minutes depending on your internet speed.*

### 🍎 For Mac / Linux Users:
1. Ensure you have Python 3 installed (`brew install python3` on Mac).
2. Download or Clone this repository.
3. Double-click the **`Install_and_Run_Mac_Linux.command`** file.
   - *If macOS says it doesn't have permission to run, open Terminal, type `chmod +x `, drag and drop the `.command` file into the terminal, and hit Enter to make it executable.*
4. A terminal window will pop up, build your virtual environment, download the AI models, and automatically open the web app in your browser!

---

## 🔑 Features & API Setup

- **Hugging Face Token:** To perform speaker separation (diarization), you need to provide a free Hugging Face API Token in the sidebar. This allows the app to securely download the Pyannote Diarization models.
- **LLM Intent Analysis:** To use the 🤖 AI Phrase Intent Analysis, select an AI Provider (OpenAI, Anthropic, or Google Gemini) and paste your API key. This will send the individual sentences to the cloud for ultra-fast, lightweight psychological classification.
- **Local Model Library:** All Whisper and Pyannote models are saved directly to your hard drive. Use the "Local Model Library" expander in the sidebar to delete old models and free up storage space.