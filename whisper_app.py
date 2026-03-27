import sys
import os
import json
import traceback
import time
import subprocess
import tempfile
import re
from datetime import datetime
import pandas as pd

PSYCH_DICT = {
    "i_talk": [r"\bi\b", r"\bme\b", r"\bmy\b", r"\bmine\b", r"\bmyself\b"],
    "you_talk": [r"\byou\b", r"\byour\b", r"\byours\b", r"\byourself\b"],
    "we_talk": [r"\bwe\b", r"\bus\b", r"\bour\b", r"\bours\b", r"\bourselves\b"],
    "criticism": [r"\balways\b", r"\bnever\b", r"\bselfish\b", r"\bcomplain\b", r"\bfault\b"],
    "contempt": [r"\bidiot\b", r"\bstupid\b", r"\bdisgusting\b", r"\bhate\b", r"\bsarcasm\b", r"\bwhatever\b"],
    "defensiveness": [r"\bbut\b", r"\bexcuse\b", r"\bwasn't me\b", r"\bnot my fault\b", r"\byou too\b"],
    "stonewalling": [r"\bfine\b", r"\bi'm done\b", r"\bleave me alone\b"],
    "restorative_positive": [r"\bappreciate\b", r"\bthank you\b", r"\bsorry\b", r"\bunderstand\b", r"\blove\b", r"\bgood\b", r"\bgreat\b", r"\bagree\b", r"\bthanks\b", r"\bhappy\b"],
    "dark_triad": [r"\bmanipulate\b", r"\blie\b", r"\btrick\b", r"\bfool\b", r"\bloser\b", r"\bweak\b", r"\bexploit\b", r"\bpathetic\b"],
    "anxious_preoccupied": [r"\bworry\b", r"\babandon\b", r"\bleave me\b", r"\bneed you\b", r"\bclingy\b", r"\bafraid\b", r"\bcare about me\b"],
    "dismissive_avoidant": [r"\bspace\b", r"\bback off\b", r"\balone\b", r"\bindependent\b", r"\bmy own\b"],
    "gaslighting": [r"\bcrazy\b", r"\bimagining things\b", r"\boverreacting\b", r"\btoo sensitive\b", r"\bparanoid\b", r"\bmade that up\b", r"\bnever happened\b", r"\bnot what i said\b"],
    "victimhood": [r"\balways me\b", r"\bmy fault\b", r"\bpoor me\b", r"\bunfair\b", r"\bpicked on\b", r"\bwhy me\b", r"\beveryone is against me\b"]
}

if "--worker" in sys.argv:
    lock_file = "jobs/worker.lock"
    lock_fd = open(lock_file, 'w')
    try:
        import fcntl
        try: fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: sys.exit(0)
    except ImportError:
        import msvcrt
        try: msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError: sys.exit(0)
        
    import whisperx
    import torch
    import gc

    def cleanup_memory():
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    while True:
        queued_jobs = []
        for filename in os.listdir("jobs"):
            if filename.endswith(".json") and not filename.endswith("_result.json"):
                try:
                    with open(os.path.join("jobs", filename), "r") as f:
                        data = json.load(f)
                        if data.get("status") == "Queued": queued_jobs.append(data)
                except: pass
        if not queued_jobs: break
            
        queued_jobs.sort(key=lambda x: x.get("id", ""))
        current_job = queued_jobs[0]
        job_id = current_job["id"]
        job_file = f"jobs/{job_id}.json"
        result_file = f"jobs/{job_id}_result.json"

        def update_status(status_text, progress=0, error=None):
            if os.path.exists(job_file):
                try:
                    with open(job_file, "r") as f: data = json.load(f)
                    data["status"] = status_text
                    data["progress"] = progress
                    if error: data["error"] = error
                    with open(job_file, "w") as f: json.dump(data, f)
                except: pass

        def calculate_audio_metrics(seg, full_audio, i, all_segments):
            if full_audio is None: return
            start_sec = seg.get("start", 0)
            end_sec = seg.get("end", 0)
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            chunk = full_audio[start_ms:end_ms]
            
            dbfs = chunk.dBFS
            if dbfs == float('-inf'): dbfs = -100.0
            
            target_dbfs = -20.0
            gain_adj = target_dbfs - dbfs
            
            window_ms = 100
            slices = [chunk[j:j+window_ms] for j in range(0, len(chunk), window_ms)]
            slice_dbfs = [s.dBFS for s in slices if s.dBFS != float('-inf')]
            
            if slice_dbfs:
                dynamic_range = max(slice_dbfs) - min(slice_dbfs)
                crest_factor = chunk.max_dBFS - dbfs
                silence_density = sum(1 for s in slices if s.dBFS < -60.0 or s.dBFS == float('-inf')) / len(slices)
            else:
                dynamic_range = 0
                crest_factor = 0
                silence_density = 0.0
                
            interruption = False
            if i > 0:
                prev_seg = all_segments[i-1]
                if prev_seg.get("speaker") != seg.get("speaker", "UNKNOWN") and start_sec < prev_seg.get("end", 0) - 0.2:
                    interruption = True

            text = seg.get("text", "").strip()
            duration_min = (end_sec - start_sec) / 60.0
            wpm = len(text.split()) / duration_min if duration_min > 0 else 0
            
            seg["dbfs"] = round(dbfs, 1)
            seg["wpm"] = round(wpm, 1)
            seg["gain_adj"] = round(gain_adj, 1)
            seg["dynamic_range"] = round(dynamic_range, 1)
            seg["crest_factor"] = round(crest_factor, 1)
            seg["silence_density"] = round(silence_density, 2)
            if "audio_flags" not in seg: seg["audio_flags"] = []
            if interruption: seg["audio_flags"].append("Interruption")

        try:
            update_status("Initializing AI engine...", 5)
            with open(job_file, "r") as f: job_data = json.load(f)

            device_opt = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = job_data.get("compute_type", "int8")
            whisper_model = job_data.get("model", "base")
            lang_code = job_data.get("language")
            hf_token = job_data.get("hf_token")
            min_speakers = job_data.get("min_speakers", 1)
            max_speakers = job_data.get("max_speakers", 3)
            audio_path = job_data["filepath"]
            
            calc_volume = job_data.get("calc_volume", False)
            calc_sentiment = job_data.get("calc_sentiment", False)
            calc_psych = job_data.get("calc_psych", False)
            calc_llm = job_data.get("calc_llm", False)
            retroactive_only = job_data.get("retroactive_only", False)
            
            out_data = None

            if retroactive_only:
                update_status("Calculating acoustic, emotional, and psychological tone...", 10)
                try:
                    full_audio = None
                    analyzer = None
                    if calc_volume:
                        import pydub
                        from pydub import AudioSegment
                        full_audio = AudioSegment.from_file(audio_path)
                    if calc_sentiment:
                        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                        analyzer = SentimentIntensityAnalyzer()
                        
                    with open(result_file, "r") as f: out_data = json.load(f)
                    segments = out_data.get("segments", [])
                    total_segs = len(segments)
                    psych_stats = {}
                    
                    for i, seg in enumerate(segments):
                        text = seg.get("text", "").strip()
                        speaker = seg.get("speaker", "UNKNOWN")
                        t_lower = text.lower()
                        
                        if calc_volume:
                            calculate_audio_metrics(seg, full_audio, i, segments)
                        if analyzer is not None:
                            vs = analyzer.polarity_scores(text)
                            compound = vs['compound']
                            if compound >= 0.05: sentiment = "Positive"
                            elif compound <= -0.05: sentiment = "Negative"
                            else: sentiment = "Neutral"
                            seg["sentiment"] = sentiment
                            seg["sentiment_score"] = compound
                        if calc_psych:
                            if speaker not in psych_stats:
                                psych_stats[speaker] = {k: 0 for k in PSYCH_DICT.keys()}
                                psych_stats[speaker].update({"total_turns": 0, "total_words": 0, "questions": 0, "sum_dbfs": 0, "sum_crest": 0, "has_audio": False})
                                
                            psych_stats[speaker]["total_turns"] += 1
                            psych_stats[speaker]["total_words"] += max(1, len(text.split()))
                            psych_stats[speaker]["questions"] += text.count("?")
                            if calc_volume and "dbfs" in seg:
                                psych_stats[speaker]["sum_dbfs"] += seg["dbfs"]
                                psych_stats[speaker]["sum_crest"] += seg.get("crest_factor", 0)
                                psych_stats[speaker]["has_audio"] = True

                            seg["psych_flags"] = []
                            for cat, patterns in PSYCH_DICT.items():
                                for p in patterns:
                                    matches = len(re.findall(p, t_lower))
                                    if matches > 0:
                                        psych_stats[speaker][cat] += matches
                                        flag_name = cat.replace("_", " ").title()
                                        if flag_name not in seg["psych_flags"]:
                                            seg["psych_flags"].append(flag_name)
                        if i % max(1, int(total_segs/20)) == 0:
                            update_status("Adding AI Metrics...", 10 + int(85 * (i/max(1, total_segs))))
                            
                    if calc_psych:
                        for spk, stats in psych_stats.items():
                            pos = stats.get("restorative_positive", 0)
                            neg = stats.get("criticism", 0) + stats.get("contempt", 0) + stats.get("defensiveness", 0) + stats.get("stonewalling", 0) + stats.get("dark_triad", 0) + stats.get("gaslighting", 0) + stats.get("victimhood", 0)
                            stats["gottman_ratio_val"] = round(pos / neg, 2) if neg > 0 else (pos if pos > 0 else 0)
                            stats["gottman_ratio_desc"] = f"{pos}:{neg}"
                            turns = max(1, stats.get("total_turns", 1))
                            words = max(1, stats.get("total_words", 1))
                            has_audio = stats.get("has_audio", False)
                            avg_dbfs = stats.get("sum_dbfs", 0) / turns if has_audio else -20.0
                            avg_crest = stats.get("sum_crest", 0) / turns if has_audio else 10.0
                            avg_words = stats.get("total_words", 0) / turns
                            we_freq = (stats.get("we_talk", 0) / words) * 100
                            you_freq = (stats.get("you_talk", 0) / words) * 100
                            i_freq = (stats.get("i_talk", 0) / words) * 100
                            repair_freq = (stats.get("restorative_positive", 0) / words) * 100
                            victim_freq = stats.get("victimhood", 0)
                            q_per_turn = stats.get("questions", 0) / turns
                            roles = []
                            if we_freq > 1.5 and repair_freq > 0.5: roles.append("🤝 Conciliator")
                            if you_freq > 2.5 and (avg_crest > 15 or avg_dbfs > -15): roles.append("⚖️ Prosecutor")
                            if avg_words < 12 and avg_crest < 12: roles.append("🧱 Stonewaller")
                            if i_freq > 3.0 and victim_freq > 0: roles.append("🥀 Victim")
                            if we_freq > 1.0 and q_per_turn > 0.3 and (-25 <= avg_dbfs <= -12): roles.append("🦉 Socratic Mentor")
                            if not roles: roles.append("Neutral")
                            stats["roles"] = roles
                        out_data["psych_stats"] = psych_stats
                except Exception as metric_err: print(f"Error calculating metrics: {metric_err}")

            else:
                engine = current_job.get("engine", "⚡ GPU/Apple Silicon (insanely-fast-whisper)")
                if "CPU" in engine:
                    update_status("Loading audio file...", 10)
                    audio = whisperx.load_audio(audio_path)
                    
                    update_status(f"Loading WhisperX model ({whisper_model})...", 15)
                    model = whisperx.load_model(whisper_model, device_opt, compute_type=compute_type)
                    
                    update_status("Transcribing audio (this takes the longest)...", 30)
                    result = model.transcribe(audio, batch_size=16, language=lang_code)
                    detected_lang = result["language"]
                    
                    del model
                    cleanup_memory()

                    update_status(f"Aligning text to audio ({detected_lang})...", 50)
                    model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device_opt)
                    result = whisperx.align(result["segments"], model_a, metadata, audio, device_opt, return_char_alignments=False)
                    
                    del model_a
                    cleanup_memory()

                    update_status("Diarizing speakers...", 70)
                    from whisperx.diarize import DiarizationPipeline
                    diarize_model = DiarizationPipeline(token=hf_token, device=device_opt)
                    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
                    
                    del diarize_model
                    cleanup_memory()

                    update_status("Assigning speakers to words...", 90)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                else:
                    import subprocess
                    update_status(f"Transcribing & Diarizing with insanely-fast-whisper ({whisper_model})...", 20)
                    
                    # Format model name (e.g. 'large-v2' to 'openai/whisper-large-v2')
                    hf_model = f"openai/whisper-{whisper_model}" if "openai" not in whisper_model else whisper_model
                    
                    dev_id = "0" if device_opt == "cuda" else "cpu"
                    if device_opt == "mps": dev_id = "mps"
                    
                    out_json = job_file.replace(".json", "_ifw.json")
                    
                    import shutil
                    ifw_bin = shutil.which("insanely-fast-whisper") or "insanely-fast-whisper"
                    cmd = [
                        ifw_bin,
                        "--file-name", audio_path,
                        "--model-name", hf_model,
                        "--transcript-path", out_json,
                        "--device-id", dev_id,
                        "--batch-size", "1" if dev_id == "cpu" else "8"
                    ]
                    
                    if lang_code:
                        cmd.extend(["--language", lang_code])
                        
                    if hf_token:
                        cmd.extend(["--hf-token", hf_token])
                    
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        raise Exception(f"insanely-fast-whisper failed: {e}")
                    
                    # Parse output to match whisperx format for downstream metrics
                    with open(out_json, "r") as f:
                        ifw_data = json.load(f)
                        
                    segments = []
                    for chunk in ifw_data.get("chunks", []):
                        ts = chunk.get("timestamp", [0.0, 0.0])
                        segments.append({
                            "start": ts[0],
                            "end": ts[1],
                            "text": chunk.get("text", ""),
                            "speaker": chunk.get("speaker", "UNKNOWN")
                        })
                        
                    result = {"segments": segments, "language": lang_code or "en"}
                    
                    try: os.remove(out_json)
                    except: pass
                    
                    update_status("Transcription complete. Applying formatting...", 90)

                if calc_volume or calc_sentiment or calc_psych:
                    update_status("Calculating acoustic, emotional, and psychological tone...", 92)
                    try:
                        full_audio = None
                        analyzer = None
                        if calc_volume:
                            import pydub
                            from pydub import AudioSegment
                            full_audio = AudioSegment.from_file(audio_path)
                        if calc_sentiment:
                            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                            analyzer = SentimentIntensityAnalyzer()
                            
                        psych_stats = {}
                        segments = result.get("segments", [])
                        
                        for i, seg in enumerate(segments):
                            text = seg.get("text", "").strip()
                            speaker = seg.get("speaker", "UNKNOWN")
                            t_lower = text.lower()
                            
                            if calc_volume:
                                calculate_audio_metrics(seg, full_audio, i, segments)
                                
                            if analyzer is not None:
                                vs = analyzer.polarity_scores(text)
                                compound = vs['compound']
                                if compound >= 0.05: sentiment = "Positive"
                                elif compound <= -0.05: sentiment = "Negative"
                                else: sentiment = "Neutral"
                                seg["sentiment"] = sentiment
                                seg["sentiment_score"] = compound
                                
                            if calc_psych:
                                if speaker not in psych_stats:
                                    psych_stats[speaker] = {k: 0 for k in PSYCH_DICT.keys()}
                                    psych_stats[speaker].update({"total_turns": 0, "total_words": 0, "questions": 0, "sum_dbfs": 0, "sum_crest": 0, "has_audio": False})
                                    
                                psych_stats[speaker]["total_turns"] += 1
                                psych_stats[speaker]["total_words"] += max(1, len(text.split()))
                                psych_stats[speaker]["questions"] += text.count("?")
                                if calc_volume and "dbfs" in seg:
                                    psych_stats[speaker]["sum_dbfs"] += seg["dbfs"]
                                    psych_stats[speaker]["sum_crest"] += seg.get("crest_factor", 0)
                                    psych_stats[speaker]["has_audio"] = True

                                seg["psych_flags"] = []
                                for cat, patterns in PSYCH_DICT.items():
                                    for p in patterns:
                                        matches = len(re.findall(p, t_lower))
                                        if matches > 0:
                                            psych_stats[speaker][cat] += matches
                                            flag_name = cat.replace("_", " ").title()
                                            if flag_name not in seg["psych_flags"]:
                                                seg["psych_flags"].append(flag_name)
                                                
                        if calc_psych:
                            for spk, stats in psych_stats.items():
                                pos = stats.get("restorative_positive", 0)
                                neg = stats.get("criticism", 0) + stats.get("contempt", 0) + stats.get("defensiveness", 0) + stats.get("stonewalling", 0) + stats.get("dark_triad", 0) + stats.get("gaslighting", 0) + stats.get("victimhood", 0)
                                stats["gottman_ratio_val"] = round(pos / neg, 2) if neg > 0 else (pos if pos > 0 else 0)
                                stats["gottman_ratio_desc"] = f"{pos}:{neg}"
                                
                                turns = max(1, stats.get("total_turns", 1))
                                words = max(1, stats.get("total_words", 1))
                                has_audio = stats.get("has_audio", False)
                                
                                avg_dbfs = stats.get("sum_dbfs", 0) / turns if has_audio else -20.0
                                avg_crest = stats.get("sum_crest", 0) / turns if has_audio else 10.0
                                avg_words = stats.get("total_words", 0) / turns
                                
                                we_freq = (stats.get("we_talk", 0) / words) * 100
                                you_freq = (stats.get("you_talk", 0) / words) * 100
                                i_freq = (stats.get("i_talk", 0) / words) * 100
                                repair_freq = (stats.get("restorative_positive", 0) / words) * 100
                                victim_freq = stats.get("victimhood", 0)
                                q_per_turn = stats.get("questions", 0) / turns
                                
                                roles = []
                                if we_freq > 1.5 and repair_freq > 0.5: roles.append("🤝 Conciliator")
                                if you_freq > 2.5 and (avg_crest > 15 or avg_dbfs > -15): roles.append("⚖️ Prosecutor")
                                if avg_words < 12 and avg_crest < 12: roles.append("🧱 Stonewaller")
                                if i_freq > 3.0 and victim_freq > 0: roles.append("🥀 Victim")
                                if we_freq > 1.0 and q_per_turn > 0.3 and (-25 <= avg_dbfs <= -12): roles.append("🦉 Socratic Mentor")
                                
                                if not roles: roles.append("Neutral")
                                stats["roles"] = roles
                                
                            result["psych_stats"] = psych_stats

                    except Exception as metric_err:
                        print(f"Error calculating metrics: {metric_err}")

                out_data = result

            if calc_llm and out_data:
                update_status("🤖 Running Local AI Phrase Analysis...", 92)
                try:
                    import requests
                    segs = out_data.get("segments", [])
                    total = len(segs)
                    api_provider = current_job.get("api_provider", "OpenAI")
                    api_key = current_job.get("api_key", "")
                    
                    if not api_key:
                        raise Exception("API key missing. Skipping LLM analysis.")
                    
                    for idx, seg in enumerate(segs):
                        text = seg.get("text", "").strip()
                        if len(text.split()) < 3:
                            seg["llm_intent"] = "Neutral"
                            continue
                        
                        prompt = f"Analyze this sentence and classify its primary psychological intent into EXACTLY ONE of these categories: [Observation, Question, Defensiveness, Hostility, Empathy/Repair, Victimhood, Gaslighting, Neutral]. Return NOTHING ELSE but the exact category name.\n\nSentence: \"{text}\""
                        ans = "Neutral"
                        try:
                            headers = {}
                            if api_provider == "OpenAI":
                                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                                data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
                                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=10)
                                if r.status_code == 200:
                                    ans = r.json()["choices"][0]["message"]["content"].strip().replace(".", "").replace('"', '')
                            
                            elif api_provider == "Anthropic":
                                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
                                data = {"model": "claude-3-haiku-20240307", "messages": [{"role": "user", "content": prompt}], "max_tokens": 10, "temperature": 0.0}
                                r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=10)
                                if r.status_code == 200:
                                    ans = r.json()["content"][0]["text"].strip().replace(".", "").replace('"', '')
                                    
                            elif api_provider == "Google Gemini":
                                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                                headers = {"Content-Type": "application/json"}
                                data = {"contents": [{"parts": [{"text": prompt}]}]}
                                r = requests.post(url, headers=headers, json=data, timeout=10)
                                if r.status_code == 200:
                                    ans = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().replace(".", "").replace('"', '')
                        except Exception as e:
                            pass
                            
                        seg["llm_intent"] = ans
                        if idx % max(1, int(total/10)) == 0:
                            update_status(f"🤖 AI Intent Analysis ({api_provider})... {int((idx/total)*100)}%", 92)
                except Exception as llm_err:
                    print(f"LLM Error: {llm_err}")

            update_status("Saving results...", 95)
            with open(result_file, "w") as f: json.dump(out_data, f, ensure_ascii=False)
            update_status("Completed", 100)
            
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "killed" in error_msg.lower():
                error_msg = "Out of Memory."
            update_status("Failed", 0, error=error_msg)
            
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
    except: pass
    sys.exit(0)

# ==========================================
# STREAMLIT UI APP
# ==========================================
import streamlit as st
import streamlit.components.v1 as components
import io
from docx import Document
from docx.shared import RGBColor, Pt
from streamlit_autorefresh import st_autorefresh

os.makedirs("uploads", exist_ok=True)
os.makedirs("jobs", exist_ok=True)

st.set_page_config(page_title="WhisperX Analysis Studio", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

PASTEL_COLORS = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#E8BAFF", "#FFBAE8", "#FFD1BA", "#E2FFBA", "#BAC8FF"]

st.markdown("""
<style>
    /* Segment Card Container */
    .segment-card { 
        padding: 12px 16px; 
        border-radius: 0.5rem; 
        background-color: rgba(128, 128, 128, 0.05); 
        margin-bottom: 12px; 
        border-left: 5px solid; 
        transition: all 0.2s ease; 
        position: relative;
    }
    
    /* Top Row: Speaker & Time */
    .speaker-badge { display: inline-block; padding: 0.25em 0.6em; font-size: 0.85em; font-weight: 700; border-radius: 0.25rem; color: #1f1f1f; margin-right: 0.5em; vertical-align: middle; }
    .timestamp { font-size: 0.85em; color: #888; font-family: monospace; vertical-align: middle; }
    
    /* Middle Row: Text */
    .spoken-text { font-size: 1.15em; margin: 8px 0; color: inherit; line-height: 1.5; }
    
    /* Bottom Row: Actions (Hidden by default, shown on hover/touch) */
    .action-row { 
        display: flex; 
        gap: 10px; 
        align-items: center; 
        margin-top: 5px;
        opacity: 0;
        transition: opacity 0.2s ease;
    }
    .segment-card:hover .action-row, .segment-card:active .action-row, .segment-card:focus-within .action-row {
        opacity: 1;
    }
    
    /* Action Buttons (Play, Stats, Edit, Add) */
    .inline-action {
        cursor: pointer; background-color: #007bff; color: white; border-radius: 4px;
        padding: 4px 10px; font-size: 0.85em; display: inline-flex; align-items: center; justify-content: center;
        user-select: none; font-weight: bold; text-decoration: none !important; border: none;
    }
    .inline-action.play { background-color: #28a745; }
    .inline-action.play:hover { background-color: #218838; }
    .inline-action.stats { background-color: #f39c12; }
    .inline-action.stats:hover { background-color: #e67e22; }
    .inline-action.edit { background-color: #6c757d; }
    .inline-action.edit:hover { background-color: #5a6268; }
    .inline-action.add { background-color: #17a2b8; }
    .inline-action.add:hover { background-color: #138496; }

    /* Colored Tags & Stats Block */
    .metrics-box { font-size: 0.85em; color: #666; background: rgba(0,0,0,0.05); padding: 8px; border-radius: 5px; margin-bottom: 8px; }
    .tag-blue { background-color: #3498db; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; margin-right: 5px; display: inline-block; margin-top: 5px; }
    .tag-green { background-color: #27ae60; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; margin-right: 5px; display: inline-block; margin-top: 5px; }
    .tag-purple { background-color: #8e44ad; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; margin-right: 5px; display: inline-block; margin-top: 5px; }
    .tag-red { background-color: #e74c3c; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; margin-right: 5px; display: inline-block; margin-top: 5px; }
    .text-red { color: #ff4b4b !important; font-weight: bold; }
    
    /* Sticky Audio Player */
    [data-testid="stAudio"] { position: fixed; bottom: 0; left: 0; width: 100%; z-index: 99999; background-color: #1e1e1e; padding: 10px 20px; box-shadow: 0px -5px 15px rgba(0,0,0,0.5); border-top: 1px solid #333; }
    .main .block-container { padding-bottom: 120px !important; }
</style>
""", unsafe_allow_html=True)

# Global JS Event Listeners & Audio Poller
components.html("""
<script>
window.parent.document.addEventListener('click', function(e) {
    if(e.target && e.target.classList.contains('inline-action')) {
        
        // Handle Play
        if(e.target.classList.contains('play')) {
            var time = e.target.getAttribute('data-time');
            var audios = window.parent.document.getElementsByTagName('audio');
            if(audios.length > 0 && time != null) {
                audios[0].currentTime = parseFloat(time);
                audios[0].play();
            }
        }
        
        // Handle Stats Toggle (Global)
        if(e.target.classList.contains('stats')) {
            var allStats = window.parent.document.querySelectorAll('.stats-row');
            var isHidden = allStats[0].style.display === 'none';
            allStats.forEach(function(el) {
                el.style.display = isHidden ? 'block' : 'none';
            });
        }
        
        // Handle Edit/Add Hooks
        if(e.target.classList.contains('hook')) {
            var hookId = e.target.getAttribute('data-hook');
            var hookBtn = window.parent.document.getElementById(hookId);
            if(hookBtn) hookBtn.click();
        }
    }
});

setInterval(function() {
    var audios = window.parent.document.getElementsByTagName('audio');
    if(audios.length > 0 && !audios[0].paused) {
        var ct = audios[0].currentTime;
        var cards = window.parent.document.querySelectorAll('.segment-card');
        cards.forEach(function(c) {
            var start = parseFloat(c.getAttribute('data-start'));
            var end = parseFloat(c.getAttribute('data-end'));
            var baseColor = c.getAttribute('data-color');
            if(ct >= start && ct <= end) {
                c.style.border = '2px solid #3498db';
                c.style.backgroundColor = 'rgba(52, 152, 219, 0.15)';
            } else {
                c.style.border = 'none';
                c.style.borderLeft = '5px solid ' + baseColor;
                c.style.backgroundColor = 'rgba(128, 128, 128, 0.05)';
            }
        });
    }
}, 250);
</script>
""", height=0, width=0)

def format_time(seconds):
    if not isinstance(seconds, (int, float)): return "00:00:00.000"
    import datetime
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

def format_srt_time(seconds): return format_time(seconds).replace('.', ',')

def get_speaker_color(speaker_id):
    try:
        if isinstance(speaker_id, str) and "_" in speaker_id:
            num = int(speaker_id.split('_')[-1])
            return PASTEL_COLORS[num % len(PASTEL_COLORS)]
    except Exception: pass
    return PASTEL_COLORS[0]

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return RGBColor(int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))

def get_all_jobs():
    jobs = []
    for filename in os.listdir("jobs"):
        if filename.endswith(".json") and not filename.endswith("_result.json"):
            try:
                with open(os.path.join("jobs", filename), "r") as f: jobs.append(json.load(f))
            except: pass
    return sorted(jobs, key=lambda x: x.get("id", ""), reverse=True)

def queue_retroactive(job_id):
    job_file = f"jobs/{job_id}.json"
    result_file = f"jobs/{job_id}_result.json"
    if not os.path.exists(job_file) or not os.path.exists(result_file): return
    with open(job_file, "r") as f: job_data = json.load(f)
    audio_path = job_data.get("filepath", "")
    new_id = f"job_{int(time.time())}"
    new_job_file = f"jobs/{new_id}.json"
    new_result_file = f"jobs/{new_id}_result.json"
    suffix = os.path.splitext(audio_path)[1]
    new_audio_path = os.path.join("uploads", f"{new_id}{suffix}")
    if os.path.exists(audio_path): os.rename(audio_path, new_audio_path)
    if os.path.exists(result_file): os.rename(result_file, new_result_file)
    job_data["id"] = new_id
    job_data["filepath"] = new_audio_path
    job_data["status"] = "Queued"
    job_data["progress"] = 0
    job_data["calc_volume"] = True
    job_data["calc_sentiment"] = True
    job_data["calc_psych"] = True
    job_data["calc_llm"] = True
    job_data["retroactive_only"] = True
    with open(new_job_file, "w") as f: json.dump(job_data, f)
    if os.path.exists(job_file): os.remove(job_file)
    subprocess.Popen([sys.executable, "whisper_app.py", "--worker"])

if 'speaker_map' not in st.session_state: st.session_state.speaker_map = {}
if 'view_job_id' not in st.session_state: st.session_state.view_job_id = None


import shutil

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
local_models = []
if os.path.exists(cache_dir):
    for f in os.listdir(cache_dir):
        if "whisper" in f.lower() and os.path.isdir(os.path.join(cache_dir, f)):
            size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(os.path.join(cache_dir, f)) for filename in filenames)
            size_mb = round(size / (1024 * 1024), 1)
            local_models.append({"folder": f, "size": size_mb})

with st.sidebar:
    st.title("⚙️ Settings")
    hf_token = st.text_input("Hugging Face Token", type="password")
    st.markdown("---")
    st.subheader("LLM API Configuration")
    api_provider = st.selectbox("AI Provider", ["OpenAI", "Anthropic", "Google Gemini"])
    api_key = st.text_input(f"{api_provider} API Key", type="password", help=f"Enter your {api_provider} API key for advanced phrase analysis.")
    
    st.markdown("---")
    with st.expander("📦 Local Model Library", expanded=False):
        st.caption("Manage downloaded models to free up disk space. (Models are automatically downloaded the first time you run them).")
        if not local_models:
            st.info("No models cached yet.")
        else:
            for m in local_models:
                # simplify name
                clean_name = m['folder'].replace("models--openai--", "").replace("models--guillaumekln--faster-", "").replace("models--Systran--faster-", "")
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"💾 **{clean_name}**  \n({m['size']} MB)")
                with c2:
                    if st.button("🗑️", key=f"del_model_{m['folder']}"):
                        shutil.rmtree(os.path.join(cache_dir, m['folder']))
                        st.rerun()
                st.markdown("---")

    st.markdown("---")
    st.subheader("Whisper Configuration")

    whisper_model = st.selectbox("Model Size", ["large-v3", "large-v2", "medium", "small", "base", "tiny"], index=4)
    compute_type = st.selectbox("Compute Type", ["int8", "float32"], index=0)
    language = st.selectbox("Language", ["Auto", "en", "es", "fr", "de", "it", "pt", "ja", "zh", "nl"], index=0)
    lang_code = None if language == "Auto" else language
    st.markdown("---")
    st.subheader("Diarization Configuration")
    col1, col2 = st.columns(2)
    with col1: min_speakers = st.number_input("Min Speakers", min_value=1, max_value=10, value=1)
    with col2: max_speakers = st.number_input("Max Speakers", min_value=1, max_value=10, value=3)

st.title("🎙️ WhisperX Pro (Analysis Studio)")

tab_jobs, tab_upload, tab_help = st.tabs(["📋 Active & Past Jobs", "📁 Start New Job", "📖 Interpreting Analysis"])

with tab_upload:
    st.subheader("Start a New Job")
    audio_file = st.file_uploader("Upload Audio/Video", type=["mp3", "wav", "m4a", "ogg", "flac", "mp4"])
    engine_opt = st.radio("Processing Engine:", ["⚡ GPU/Apple Silicon (insanely-fast-whisper)", "💻 CPU Optimized (whisperX)"])
    st.markdown("#### Select Analysis Tools")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.checkbox("📝 Transcription & Diarization", value=True, disabled=True)
        calc_volume = st.checkbox("🔊 Advanced Acoustic Analysis", value=True)
    with col_t2:
        calc_sentiment = st.checkbox("🎭 Emotional Tone (VADER)", value=True)
        calc_psych = st.checkbox("🧠 Psychological Word Scan", value=True)
        calc_llm = st.checkbox("🤖 AI Phrase Intent (Qwen 2.5)", value=True, help="Uses local Qwen 1.5B LLM to logically classify the sentence's intent.")

    if st.button("🚀 Start Background Job", type="primary"):
        if not audio_file: st.error("⚠️ Please upload an audio file first.")
        elif not hf_token: st.error("⚠️ Hugging Face token is required.")
        else:
            job_id = f"job_{int(time.time())}"
            suffix = os.path.splitext(audio_file.name)[1]
            safe_filepath = os.path.join("uploads", f"{job_id}{suffix}")
            with open(safe_filepath, "wb") as f:
                if hasattr(audio_file, "getvalue"): f.write(audio_file.getvalue())
                else: f.write(audio_file.read())
            job_data = {
                "id": job_id, "filename": audio_file.name, "filepath": safe_filepath,
                "model": whisper_model, "compute_type": compute_type, "hf_token": hf_token,
                "language": lang_code, "min_speakers": min_speakers, "max_speakers": max_speakers,
                "calc_volume": calc_volume, "calc_sentiment": calc_sentiment, "calc_psych": calc_psych, "calc_llm": calc_llm,
                "api_provider": api_provider,
                "api_key": api_key,
                "engine": engine_opt,
                "status": "Queued", "progress": 0, "error": ""
            }
            with open(f"jobs/{job_id}.json", "w") as f: json.dump(job_data, f)
            subprocess.Popen([sys.executable, "whisper_app.py", "--worker"])
            st.success("Added to queue!")
            time.sleep(1)
            st.rerun()

with tab_jobs:
    jobs = get_all_jobs()
    has_active_jobs = any(j.get("status") not in ["Completed", "Failed"] for j in jobs)
    
    col_r1, col_r2 = st.columns([8, 2])
    with col_r1: st.subheader("Job Queue")
    with col_r2:
        if st.toggle("Auto-Refresh", value=has_active_jobs): st_autorefresh(interval=3000, limit=None, key="job_refresh")
    
    for job in jobs:
        with st.container():
            col_info, col_actions = st.columns([6, 4])
            with col_info:
                st.write(f"**File:** {job['filename']}")
                tools_str = "📝"
                if job.get('calc_volume', False): tools_str += " 🔊"
                if job.get('calc_sentiment', False): tools_str += " 🎭"
                if job.get('calc_psych', False): tools_str += " 🧠"
                if job.get('calc_llm', False): tools_str += " 🤖"
                st.caption(f"ID: {job['id']} | Model: {job.get('model', 'base')} | Tools: {tools_str}")
                
                if job['status'] == "Failed": st.error("Failed")
                elif job['status'] == "Completed": pass
                elif job['status'] == "Queued": st.info("⏱️ Queued")
                else: st.progress(job['progress'], text=f"{job['status']} ({job['progress']}%)")
                
            with col_actions:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                b1, b2, b3 = st.columns([1,1,1])
                if job['status'] == "Completed":
                    with b1:
                        if st.button("👁️ View", key=f"view_{job['id']}"):
                            st.session_state.view_job_id = job['id']
                            st.rerun()
                    with b2:
                        if not (job.get('calc_volume') and job.get('calc_sentiment') and job.get('calc_psych') and job.get('calc_llm')):
                            if st.button("✨ Add AI", key=f"retro_{job['id']}"):
                                queue_retroactive(job['id'])
                                st.rerun()
                with b3:
                    if st.button("🗑 Delete", key=f"del_{job['id']}"):
                        try: os.remove(job['filepath'])
                        except: pass
                        try: os.remove(f"jobs/{job['id']}.json")
                        except: pass
                        try: os.remove(f"jobs/{job['id']}_result.json")
                        except: pass
                        st.rerun()
            st.markdown("---")

with tab_help:
    st.markdown("## 📖 Interpreting Analysis Metrics")

if st.session_state.view_job_id is not None:
    job_id = st.session_state.view_job_id
    result_file = f"jobs/{job_id}_result.json"
    job_file = f"jobs/{job_id}.json"
    
    if os.path.exists(result_file):
        with open(result_file, "r") as f: res = json.load(f)
        job_meta = {}
        if os.path.exists(job_file):
            with open(job_file, "r") as f: job_meta = json.load(f)
            
        segments = res.get("segments", [])
        psych_stats = res.get("psych_stats", {})
        
        st.header(f"Results for Job: {job_id}")
        
        if os.path.exists(job_meta.get("filepath", "")):
            st.audio(job_meta.get("filepath"))
        
        tools_used = ["📝 WhisperX"]
        if job_meta.get("calc_volume", False) or any("dbfs" in s for s in segments[:5]): tools_used.append("🔊 Acoustic Analysis")
        if job_meta.get("calc_sentiment", False) or any("sentiment" in s for s in segments[:5]): tools_used.append("🎭 VADER")
        if job_meta.get("calc_psych", False) or psych_stats: tools_used.append("🧠 Psych Scanning")
        if job_meta.get("calc_llm", False) or any("llm_intent" in s for s in segments[:5]): tools_used.append("🤖 Qwen AI Intent")
            
        st.markdown(f"<div class='tools-used'><b>Tools Applied:</b> {' &nbsp;|&nbsp; '.join(tools_used)}</div>", unsafe_allow_html=True)
        
        if st.button("← Back to Job List"):
            st.session_state.view_job_id = None
            st.rerun()
            
        speakers = sorted(list(set([s.get("speaker", "UNKNOWN") for s in segments])))
        cols = st.columns(min(len(speakers) if speakers else 1, 4))
        for i, spk in enumerate(speakers):
            with cols[i % 4]:
                color = get_speaker_color(spk)
                st.markdown(f'<span style="background-color: {color}; padding: 3px 8px; border-radius: 4px; font-weight: bold; color: #000;">{spk}</span>', unsafe_allow_html=True)
                st.session_state.speaker_map[spk] = st.text_input(f"Rename", value=st.session_state.speaker_map.get(spk, spk), key=f"rename_{job_id}_{spk}", label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 🎛️ Analysis Thresholds")
        col_th1, col_th2, col_th3, col_th4 = st.columns(4)
        with col_th1: dr_thresh = st.number_input("🔴 High DR Threshold", value=25, step=1)
        with col_th2: crest_thresh = st.number_input("🔴 High Crest Threshold", value=20, step=1)
        with col_th3: whisper_thresh = st.number_input("🗣️ Whisper Threshold (Gain > X)", value=15, step=1)
        with col_th4: shout_thresh = st.number_input("📢 Shout Threshold (Gain < X)", value=-5, step=1)
        st.markdown("---")
        
        speaker_whispers = {spk: 0 for spk in speakers}
        speaker_shouts = {spk: 0 for spk in speakers}
        vader_stats = {spk: {"pos":0, "neg":0, "neu":0, "total":0} for spk in speakers}
        speaker_times = {spk: 0.0 for spk in speakers}
        total_time = 0.0
        
        for s in segments:
            spk = s.get("speaker", "UNKNOWN")
            dur = s.get("end", 0) - s.get("start", 0)
            speaker_times[spk] += dur
            total_time += dur
            gain = s.get("gain_adj", 0)
            if gain > whisper_thresh: speaker_whispers[spk] += 1
            if gain < shout_thresh: speaker_shouts[spk] += 1
            sent = s.get("sentiment")
            if sent == "Positive": vader_stats[spk]["pos"] += 1
            elif sent == "Negative": vader_stats[spk]["neg"] += 1
            elif sent == "Neutral": vader_stats[spk]["neu"] += 1
            vader_stats[spk]["total"] += 1
            
        tabs = st.tabs(["📝 Transcript", "🧠 Psychological Stats", "📊 Raw Data", "💾 Downloads"])
        
        with tabs[0]:

            # Global toggle logic handled by JS
            st.markdown("<a class='inline-action stats' style='margin-bottom: 15px;'>📊 Toggle All Stats & Tags</a>", unsafe_allow_html=True)
            
            for i, seg in enumerate(segments):
                spk_id = seg.get("speaker", "UNKNOWN")
                spk_name = st.session_state.speaker_map.get(spk_id, spk_id)
                color = get_speaker_color(spk_id)
                start_sec = seg.get('start', 0)
                end_sec = seg.get('end', 0)
                start_time = format_time(start_sec)
                end_time = format_time(end_sec)
                
                # Check states
                is_editing = st.session_state.get(f"active_edit_{job_id}") == i
                is_adding = st.session_state.get(f"active_add_{job_id}") == i
                is_deleting = st.session_state.get(f"active_del_{job_id}") == i
                
                # --- EDIT MODE ---
                if is_editing:
                    with st.container():
                        st.markdown(f"<div style='border-left: 5px solid #6c757d; padding: 10px; background: rgba(0,0,0,0.05); border-radius: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                        st.markdown("**⚙️ Edit Block**")
                        col_e1, col_e2 = st.columns([1, 3])
                        with col_e1:
                            new_spk = st.selectbox("Speaker", options=speakers, index=speakers.index(spk_id) if spk_id in speakers else 0, key=f"edit_spk_{i}_{job_id}")
                        with col_e2:
                            new_text = st.text_area("Text", value=seg.get("text", ""), key=f"edit_txt_{i}_{job_id}")
                            
                        col_a1, col_a2 = st.columns(2)
                        with col_a1:
                            if st.button("💾 Save Changes", key=f"save_edit_{i}_{job_id}", type="primary"):
                                segments[i]["speaker"] = new_spk
                                segments[i]["text"] = new_text
                                res["segments"] = segments
                                with open(result_file, "w") as f: json.dump(res, f, ensure_ascii=False)
                                st.session_state[f"active_edit_{job_id}"] = None
                                st.rerun()
                        with col_a2:
                            if st.button("❌ Cancel", key=f"cancel_edit_{i}_{job_id}"):
                                st.session_state[f"active_edit_{job_id}"] = None
                                st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                    continue # Skip rendering the normal card
                
                # --- NORMAL RENDER MODE ---
                m_parts = []
                blue_tags, green_tags, purple_tags, red_tags = [], [], [], []
                
                if "dbfs" in seg: 
                    gain = seg.get('gain_adj',0)
                    m_parts.append(f"🔊 {seg['dbfs']}dB (Gain: {gain:+g}dB)")
                    dr = seg.get('dynamic_range',0)
                    m_parts.append(f"📈 DR: <span class='{'text-red' if dr > dr_thresh else ''}'>{dr}dB</span>")
                    crest = seg.get('crest_factor',0)
                    m_parts.append(f"🏔️ Crest: <span class='{'text-red' if crest > crest_thresh else ''}'>{crest}dB</span>")
                    m_parts.append(f"🔇 Sil: {int(seg.get('silence_density',0)*100)}%")
                    wpm = seg.get('wpm', 0)
                    m_parts.append(f"⏱️ <span class='{'text-red' if wpm > 220 else ''}'>{wpm} wpm</span>")
                    if gain > whisper_thresh: blue_tags.append("Whisper")
                    if gain < shout_thresh: blue_tags.append("Shout")
                    
                if "sentiment" in seg: 
                    sent = seg['sentiment']
                    m_parts.append(f"🎭 <span class='{'text-red' if sent == 'Negative' else ''}'>{sent}</span>")
                    
                if "llm_intent" in seg:
                    intent = seg["llm_intent"]
                    color_class = "text-red" if intent in ["Hostility", "Defensiveness", "Victimhood", "Gaslighting"] else ""
                    m_parts.append(f"🤖 Intent: <span class='{color_class}'>{intent}</span>")
                
                metrics_html = f"<div class='metrics-box'>{' | '.join(m_parts)}</div>" if m_parts else ""
                
                for f in seg.get("audio_flags", []): blue_tags.append(f)
                for f in seg.get("psych_flags", []):
                    if f in ["We Talk", "Restorative Positive"]: green_tags.append(f)
                    elif f in ["You Talk", "Criticism", "Contempt", "Defensiveness", "Stonewalling", "Gaslighting", "Victimhood"]: red_tags.append(f)
                    else: purple_tags.append(f)
                
                flags_html = ""
                for t in blue_tags: flags_html += f"<span class='tag-blue'>{t}</span>"
                for t in green_tags: flags_html += f"<span class='tag-green'>{t}</span>"
                for t in purple_tags: flags_html += f"<span class='tag-purple'>{t}</span>"
                for t in red_tags: flags_html += f"<span class='tag-red'>{t}</span>"
                
                header_html = f"""<div>
                        <span class="speaker-badge" style="background-color: {color};">{spk_name}</span>
                        <span class="timestamp">[{start_time} - {end_time}]</span>
                    </div>"""
                    
                if seg.get("is_note", False):
                    header_html = f"""<div><span class="speaker-badge" style="background-color: #f1c40f;">📌 NOTE</span></div>"""
                    color = "#f1c40f"

                card_html = f"""<div class="segment-card" data-start="{start_sec}" data-end="{end_sec}" data-color="{color}">
                    {header_html}
                    <div class='spoken-text'>{seg.get('text', '')}</div>
                    <div class="action-row">
                        <a class="inline-action play" data-time="{start_sec}">▶ Play</a>
                        <a class="inline-action stats">📊 Stats</a>
                        <a class="inline-action edit hook" data-hook="btn_edit_{i}_{job_id}">⚙️ Edit</a>
                        <a class="inline-action add hook" data-hook="btn_add_{i}_{job_id}">➕ Note</a>
                        <a class="inline-action hook" data-hook="btn_del_{i}_{job_id}" style="background-color: #dc3545;">🗑 Delete</a>
                    </div>
                    <div class="stats-row" style="display:none; margin-top: 10px;">
                        {metrics_html}
                        <div style="margin-top: 5px;">{flags_html}</div>
                    </div>
                </div>"""
                
                st.markdown(card_html, unsafe_allow_html=True)

                # Invisible hooks
                with st.container():
                    st.markdown("<div style='display:none;'>", unsafe_allow_html=True)
                    if st.button("edit", key=f"btn_edit_{i}_{job_id}"):
                        st.session_state[f"active_edit_{job_id}"] = i
                        st.session_state[f"active_add_{job_id}"] = None
                        st.session_state[f"active_del_{job_id}"] = None
                        st.rerun()
                    if st.button("add", key=f"btn_add_{i}_{job_id}"):
                        st.session_state[f"active_add_{job_id}"] = i
                        st.session_state[f"active_edit_{job_id}"] = None
                        st.session_state[f"active_del_{job_id}"] = None
                        st.rerun()
                    if st.button("del", key=f"btn_del_{i}_{job_id}"):
                        st.session_state[f"active_del_{job_id}"] = i
                        st.session_state[f"active_edit_{job_id}"] = None
                        st.session_state[f"active_add_{job_id}"] = None
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                # --- DELETE CONFIRMATION ---
                if is_deleting:
                    st.warning("⚠️ Are you sure you want to permanently delete this block?")
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        if st.button("🗑️ Yes, Delete", key=f"confirm_del_{i}_{job_id}", type="primary"):
                            segments.pop(i)
                            res["segments"] = segments
                            with open(result_file, "w") as f: json.dump(res, f, ensure_ascii=False)
                            st.session_state[f"active_del_{job_id}"] = None
                            st.rerun()
                    with col_d2:
                        if st.button("❌ Cancel", key=f"cancel_del_{i}_{job_id}"):
                            st.session_state[f"active_del_{job_id}"] = None
                            st.rerun()

                # --- ADD BLOCK MODE ---            
                if is_adding:
                    st.markdown(f"<div style='border-left: 5px solid #17a2b8; padding: 10px; background: rgba(0,0,0,0.05); border-radius: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    st.markdown(f"**➕ Add Block Beneath**")
                    
                    block_type = st.radio("Block Type", options=["🗣️ Spoken Dialogue (with timestamp & speaker)", "📌 Analyst Note (no speaker/time)"], key=f"type_add_{i}_{job_id}")
                    
                    if "Spoken" in block_type:
                        col_a1, col_a2 = st.columns(2)
                        with col_a1:
                            new_end = st.number_input("End Timestamp (Seconds)", value=end_sec + 2.0, step=0.5, key=f"end_add_{i}_{job_id}")
                        with col_a2:
                            note_spk = st.selectbox("Speaker", options=speakers, key=f"spk_add_{i}_{job_id}")
                    
                    new_text = st.text_area("Content", key=f"txt_add_{i}_{job_id}")
                        
                    col_f1, col_f2 = st.columns(2)
                    with col_f1: 
                        if st.button("💾 Save Block", key=f"save_add_{i}_{job_id}", type="primary"):
                            if new_text.strip():
                                if "Spoken" in block_type:
                                    new_seg = {
                                        "start": end_sec + 0.001, "end": new_end,
                                        "speaker": note_spk,
                                        "text": new_text.strip()
                                    }
                                    try:
                                        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                                        vs = SentimentIntensityAnalyzer().polarity_scores(new_text)
                                        if vs['compound'] >= 0.05: new_seg["sentiment"] = "Positive"
                                        elif vs['compound'] <= -0.05: new_seg["sentiment"] = "Negative"
                                        else: new_seg["sentiment"] = "Neutral"
                                    except: pass
                                    new_seg["psych_flags"] = []
                                    t_lower = new_text.lower()
                                    for cat, patterns in PSYCH_DICT.items():
                                        for p in patterns:
                                            if re.search(p, t_lower):
                                                flag_name = cat.replace("_", " ").title()
                                                if flag_name not in new_seg["psych_flags"]:
                                                    new_seg["psych_flags"].append(flag_name)
                                else:
                                    new_seg = {
                                        "is_note": True,
                                        "text": new_text.strip()
                                    }
                                    
                                segments.insert(i+1, new_seg)
                                res["segments"] = segments
                                with open(result_file, "w") as f: json.dump(res, f, ensure_ascii=False)
                                st.session_state[f"active_add_{job_id}"] = None
                                st.rerun()
                    with col_f2:
                        if st.button("❌ Cancel", key=f"cancel_add_{i}_{job_id}"):
                            st.session_state[f"active_add_{job_id}"] = None
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        with tabs[1]:
            if not psych_stats:
                st.info("No psychological stats available. Click '✨ Add Analysis' to calculate them.")
            else:
                st.subheader("Gottman Ratios & Keyword Frequencies")
                st.markdown("Ratio of Positive statements to Negative statements (Criticism, Contempt, Defensiveness, Stonewalling, Dark Triad, Gaslighting, Victimhood). Stable relationships target a 5:1 ratio.")
                
                for spk, stats in psych_stats.items():
                    spk_name = st.session_state.speaker_map.get(spk, spk)
                    spk_time = speaker_times.get(spk, 0)
                    spk_pct = (spk_time / total_time * 100) if total_time > 0 else 0
                    time_str = f"⏱️ Speaking Time: {spk_time/60:.1f} mins ({spk_pct:.1f}%)"
                    
                    with st.expander(f"📊 {spk_name} | Gottman Ratio: {stats.get('gottman_ratio_desc', '0:0')} = {stats.get('gottman_ratio_val', 0)} | {time_str}", expanded=True):
                        st.markdown(f"**🎭 Roles Detected:** `{'` `'.join(stats.get('roles', ['Neutral']))}`")
                        
                        v_stat = vader_stats.get(spk, {"pos":0, "neg":0, "neu":0, "total":0})
                        v_tot = max(v_stat["total"], 1)
                        w_count = speaker_whispers.get(spk, 0)
                        s_count = speaker_shouts.get(spk, 0)
                        
                        st.markdown(f"**🧠 VADER Sentiment Overall:** {v_stat['pos']} Pos ({v_stat['pos']/v_tot*100:.0f}%) | {v_stat['neg']} Neg ({v_stat['neg']/v_tot*100:.0f}%) | {v_stat['neu']} Neu ({v_stat['neu']/v_tot*100:.0f}%)")
                        st.markdown(f"**🔊 Detected Voice Overrides:** 🗣️ {w_count} Whispers | 📢 {s_count} Shouts")
                        st.markdown("---")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.write("**Communication Style**")
                            st.write(f"- I-Talk: {stats.get('i_talk', 0)}")
                            st.write(f"- You-Talk: {stats.get('you_talk', 0)}")
                            st.write(f"- We-Talk: {stats.get('we_talk', 0)}")
                        with col_b:
                            st.write("**Gottman Dynamics**")
                            st.write(f"✅ Restorative/Positive: {stats.get('restorative_positive', 0)}")
                            st.write(f"❌ Criticism: {stats.get('criticism', 0)}")
                            st.write(f"❌ Contempt: {stats.get('contempt', 0)}")
                            st.write(f"❌ Defensiveness: {stats.get('defensiveness', 0)}")
                            st.write(f"❌ Stonewalling: {stats.get('stonewalling', 0)}")
                        with col_c:
                            st.write("**Psychological Markers**")
                            st.write(f"🎭 Dark Triad: {stats.get('dark_triad', 0)}")
                            st.write(f"🔥 Gaslighting: {stats.get('gaslighting', 0)}")
                            st.write(f"🥀 Victimhood: {stats.get('victimhood', 0)}")
                            st.write(f"🔗 Anxious-Preoccupied: {stats.get('anxious_preoccupied', 0)}")
                            st.write(f"🔗 Dismissive-Avoidant: {stats.get('dismissive_avoidant', 0)}")

        with tabs[2]:
            df_data = []
            for seg in segments:
                row = { "Speaker": st.session_state.speaker_map.get(seg.get("speaker", "UNKNOWN"), "UNKNOWN"), "Start": seg.get("start", 0), "End": seg.get("end", 0), "Text": seg.get("text", "").strip() }
                if "dbfs" in seg: 
                    row["dBFS"] = seg["dbfs"]
                    row["Gain Adj"] = seg.get("gain_adj")
                    row["Dyn. Range"] = seg.get("dynamic_range")
                    row["Crest Fact."] = seg.get("crest_factor")
                    row["Sil. Dens."] = seg.get("silence_density")
                if "sentiment" in seg: row["Sentiment"] = seg["sentiment"]
                if "audio_flags" in seg and seg["audio_flags"]: row["Audio Flags"] = ", ".join(seg["audio_flags"])
                if "psych_flags" in seg and seg["psych_flags"]: row["Psych Flags"] = ", ".join(seg["psych_flags"])
                df_data.append(row)
            if df_data: st.dataframe(pd.DataFrame(df_data), use_container_width=True)

        with tabs[3]:
            st.subheader("Export Results")
            orig_name = os.path.splitext(job_meta.get("filename", "transcript"))[0]
            suffix = "_analyzed" if (job_meta.get("calc_volume") or job_meta.get("calc_sentiment") or job_meta.get("calc_psych")) else "_transcribed"
            export_base_name = f"{orig_name}{suffix}"
            
            txt_out = []
            if psych_stats:
                txt_out.append("=========================================")
                txt_out.append("PSYCHOLOGICAL ANALYSIS SUMMARY")
                txt_out.append("=========================================\n")
                
                for spk, stats in psych_stats.items():
                    spk_name = st.session_state.speaker_map.get(spk, spk)
                    spk_pct = (speaker_times.get(spk, 0) / total_time * 100) if total_time > 0 else 0
                    txt_out.append(f"SPEAKER: {spk_name} (Speaking Time: {spk_pct:.1f}%)")
                    txt_out.append(f"Roles Detected: {', '.join(stats.get('roles', ['Neutral']))}")
                    txt_out.append(f"Gottman Ratio (Pos:Neg): {stats.get('gottman_ratio_desc', '0:0')} = {stats.get('gottman_ratio_val', 0)}")
                    txt_out.append(f"Language: I-Talk ({stats.get('i_talk',0)}), You-Talk ({stats.get('you_talk',0)}), We-Talk ({stats.get('we_talk',0)})")
                    txt_out.append(f"Horsemen: Criticism ({stats.get('criticism',0)}), Contempt ({stats.get('contempt',0)}), Defensiveness ({stats.get('defensiveness',0)}), Stonewalling ({stats.get('stonewalling',0)})")
                    txt_out.append(f"Other: Dark Triad ({stats.get('dark_triad',0)}), Gaslighting ({stats.get('gaslighting',0)}), Victimhood ({stats.get('victimhood',0)}), Anxious ({stats.get('anxious_preoccupied',0)}), Avoidant ({stats.get('dismissive_avoidant',0)})")
                    txt_out.append("-" * 40)
                txt_out.append("\n\n")
                
            for seg in segments:
                spk = st.session_state.speaker_map.get(seg.get('speaker', 'UNKNOWN'), seg.get('speaker', 'UNKNOWN'))
                meta = ""
                m = []
                if "dbfs" in seg: m.append(f"{seg['dbfs']}dB (Gain {seg.get('gain_adj',0):+g})")
                if "sentiment" in seg: m.append(seg["sentiment"])
                if m: meta = f" [{', '.join(m)}]"
                
                flags = ""
                all_flags = seg.get('audio_flags', []) + seg.get('psych_flags', [])
                if all_flags: flags = f" <{', '.join(all_flags)}>"
                
                txt_out.append(f"[{format_time(seg.get('start', 0))}] {spk}{meta}: {seg.get('text', '').strip()}{flags}")
                
            txt_content = "\n".join(txt_out)
            
            srt_out = []
            for i, seg in enumerate(segments):
                spk = st.session_state.speaker_map.get(seg.get('speaker', 'UNKNOWN'), seg.get('speaker', 'UNKNOWN'))
                start_str = format_srt_time(seg.get('start', 0))
                end_str = format_srt_time(seg.get('end', 0))
                text = seg.get('text', '').strip()
                srt_out.append(f"{i+1}\n{start_str} --> {end_str}\n[{spk}] {text}\n")
            srt_content = "\n".join(srt_out)
            
            col_d1, col_d2 = st.columns(2)
            with col_d1: st.download_button("📄 Download TXT", data=txt_content, file_name=f"{export_base_name}.txt", mime="text/plain", use_container_width=True)
            with col_d2: st.download_button("🎬 Download SRT", data=srt_content, file_name=f"{export_base_name}.srt", mime="text/plain", use_container_width=True)
