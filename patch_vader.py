import re
import sys
import os

filepath = "/home/gingerbin/.openclaw/workspace/projects/whisper_standalone/whisper_app.py"

with open(filepath, "r") as f:
    content = f.read()

# We want to inject the Vader and PyDub logic right after whisperx.assign_word_speakers
injection_point = "result = whisperx.assign_word_speakers(diarize_segments, result)"

vader_logic = """result = whisperx.assign_word_speakers(diarize_segments, result)

            update_status("Calculating acoustic and emotional tone...", 92)
            try:
                import pydub
                from pydub import AudioSegment
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                
                analyzer = SentimentIntensityAnalyzer()
                # Load the full audio for slicing
                full_audio = AudioSegment.from_file(audio_path)
                
                for seg in result.get("segments", []):
                    start_sec = seg.get("start", 0)
                    end_sec = seg.get("end", 0)
                    text = seg.get("text", "").strip()
                    
                    # 1. Volume (dBFS)
                    start_ms = int(start_sec * 1000)
                    end_ms = int(end_sec * 1000)
                    chunk = full_audio[start_ms:end_ms]
                    dbfs = chunk.dBFS
                    if dbfs == float('-inf'):
                        dbfs = -100.0
                        
                    # 2. Speech Rate (WPM)
                    duration_min = (end_sec - start_sec) / 60.0
                    word_count = len(text.split())
                    wpm = word_count / duration_min if duration_min > 0 else 0
                    
                    # 3. Sentiment (VADER)
                    vs = analyzer.polarity_scores(text)
                    compound = vs['compound']
                    if compound >= 0.05:
                        sentiment = "Positive"
                    elif compound <= -0.05:
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"
                        
                    seg["dbfs"] = round(dbfs, 1)
                    seg["wpm"] = round(wpm, 1)
                    seg["sentiment"] = sentiment
                    seg["sentiment_score"] = compound
            except Exception as metric_err:
                print(f"Error calculating metrics: {metric_err}")
"""

# Replace the injection point
if injection_point in content:
    content = content.replace(injection_point, vader_logic)
    print("Injected Vader/PyDub logic.")
else:
    print("Could not find injection point!")

# Now we need to update the UI to show these metrics
ui_injection_point = """<div class="segment-card" style="border-left-color: {color};">
                    <span class="speaker-badge" style="background-color: {color};">{spk_name}</span>
                    <span class="timestamp">[{start_time} - {end_time}]</span><br>
                    <div style='margin-top:0.5rem; font-size:1.1em;'>{seg.get('text', '')}</div>
                </div>"""

ui_vader_logic = """<div class="segment-card" style="border-left-color: {color};">
                    <span class="speaker-badge" style="background-color: {color};">{spk_name}</span>
                    <span class="timestamp">[{start_time} - {end_time}]</span>
                    <span style="float: right; font-size: 0.8em; color: #666;">
                        🔊 {seg.get('dbfs', 'N/A')} dB | ⏱️ {seg.get('wpm', 'N/A')} wpm | 🎭 {seg.get('sentiment', 'N/A')}
                    </span><br>
                    <div style='margin-top:0.5rem; font-size:1.1em;'>{seg.get('text', '')}</div>
                </div>"""

if ui_injection_point in content:
    content = content.replace(ui_injection_point, ui_vader_logic)
    print("Injected UI logic.")
else:
    print("Could not find UI injection point!")

with open(filepath, "w") as f:
    f.write(content)
