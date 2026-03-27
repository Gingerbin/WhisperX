import re
with open('/home/gingerbin/.openclaw/workspace/projects/whisper_standalone/whisper_app.py', 'r') as f:
    code = f.read()

# I will replace the apply_retroactive_metrics function with queue_retroactive
def_apply = """def apply_retroactive_metrics(job_id):
    job_file = f"jobs/{job_id}.json"
    result_file = f"jobs/{job_id}_result.json"
    if not os.path.exists(job_file) or not os.path.exists(result_file): return
    
    with open(job_file, "r") as f: job_data = json.load(f)
    with open(result_file, "r") as f: res_data = json.load(f)
    
    audio_path = job_data.get("filepath", "")
    if not os.path.exists(audio_path): return
    
    try:
        import pydub
        from pydub import AudioSegment
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        full_audio = AudioSegment.from_file(audio_path)
        analyzer = SentimentIntensityAnalyzer()
        
        for seg in res_data.get("segments", []):
            start_sec = seg.get("start", 0)
            end_sec = seg.get("end", 0)
            text = seg.get("text", "").strip()
            
            # Volume & Speech Rate
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            chunk = full_audio[start_ms:end_ms]
            dbfs = chunk.dBFS
            if dbfs == float('-inf'): dbfs = -100.0
            
            duration_min = (end_sec - start_sec) / 60.0
            word_count = len(text.split())
            wpm = word_count / duration_min if duration_min > 0 else 0
            
            seg["dbfs"] = round(dbfs, 1)
            seg["wpm"] = round(wpm, 1)
            
            # Sentiment
            vs = analyzer.polarity_scores(text)
            compound = vs['compound']
            if compound >= 0.05: sentiment = "Positive"
            elif compound <= -0.05: sentiment = "Negative"
            else: sentiment = "Neutral"
            
            seg["sentiment"] = sentiment
            seg["sentiment_score"] = compound
            
        with open(result_file, "w") as f: json.dump(res_data, f, ensure_ascii=False)
        
        job_data["calc_volume"] = True
        job_data["calc_sentiment"] = True
        with open(job_file, "w") as f: json.dump(job_data, f)
        
    except Exception as e:
        print(f"Retroactive error: {e}")"""

new_queue_fn = """def queue_retroactive(job_id):
    job_file = f"jobs/{job_id}.json"
    result_file = f"jobs/{job_id}_result.json"
    if not os.path.exists(job_file) or not os.path.exists(result_file): return
    
    with open(job_file, "r") as f: job_data = json.load(f)
    audio_path = job_data.get("filepath", "")
    
    # 1. Generate new ID to push to top of list
    new_id = f"job_{int(time.time())}"
    new_job_file = f"jobs/{new_id}.json"
    new_result_file = f"jobs/{new_id}_result.json"
    
    suffix = os.path.splitext(audio_path)[1]
    new_audio_path = os.path.join("uploads", f"{new_id}{suffix}")
    
    # Rename the files
    if os.path.exists(audio_path): os.rename(audio_path, new_audio_path)
    if os.path.exists(result_file): os.rename(result_file, new_result_file)
    
    # Update job data
    job_data["id"] = new_id
    job_data["filepath"] = new_audio_path
    job_data["status"] = "Queued"
    job_data["progress"] = 0
    job_data["calc_volume"] = True
    job_data["calc_sentiment"] = True
    job_data["retroactive_only"] = True
    
    with open(new_job_file, "w") as f: json.dump(job_data, f)
    
    # Remove old JSON
    if os.path.exists(job_file): os.remove(job_file)
    
    import subprocess
    subprocess.Popen([sys.executable, "whisper_app.py", "--worker"])"""

code = code.replace(def_apply, new_queue_fn)

# I also need to update the UI loop.
ui_old = """            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
            with col1:
                st.write(f"**File:** {job['filename']}")
                tools_str = "📝"
                if job.get('calc_volume', False): tools_str += " 🔊"
                if job.get('calc_sentiment', False): tools_str += " 🎭"
                st.caption(f"ID: {job['id']} | Tools: {tools_str}")
            with col2:
                if job['status'] == "Failed": st.error("Failed")
                elif job['status'] == "Completed": st.success("Completed 100%")
                elif job['status'] == "Queued": st.info("⏱️ Queued")
                else: st.progress(job['progress'], text=f"{job['status']} ({job['progress']}%)")
            with col3:
                if job['status'] == "Completed":
                    if st.button("👁️ View Results", key=f"view_{job['id']}", type="primary"):
                        st.session_state.view_job_id = job['id']
                        st.rerun()
            with col4:
                if job['status'] == "Completed" and not (job.get('calc_volume') and job.get('calc_sentiment')):
                    if st.button("✨ Add AI Metrics", key=f"retro_{job['id']}", help="Retroactively add Volume & Sentiment"):
                        with st.spinner("Analyzing..."):
                            apply_retroactive_metrics(job['id'])
                        st.rerun()
            with col5:
                if st.button("🗑", key=f"del_{job['id']}"):
                    try: os.remove(job['filepath'])
                    except: pass
                    try: os.remove(f"jobs/{job['id']}.json")
                    except: pass
                    try: os.remove(f"jobs/{job['id']}_result.json")
                    except: pass
                    st.rerun()"""

ui_new = """            col1, col2, col3 = st.columns([4, 4, 4])
            with col1:
                st.write(f"**File:** {job['filename']}")
                tools_str = "📝"
                if job.get('calc_volume', False): tools_str += " 🔊"
                if job.get('calc_sentiment', False): tools_str += " 🎭"
                st.caption(f"ID: {job['id']} | Tools: {tools_str}")
            with col2:
                if job['status'] == "Failed": st.error("Failed")
                elif job['status'] == "Completed": st.success("Completed 100%")
                elif job['status'] == "Queued": st.info("⏱️ Queued")
                else: st.progress(job['progress'], text=f"{job['status']} ({job['progress']}%)")
            with col3:
                b1, b2, b3 = st.columns([1, 1, 1])
                with b1:
                    if job['status'] == "Completed":
                        if st.button("👁️ View", key=f"view_{job['id']}", type="primary"):
                            st.session_state.view_job_id = job['id']
                            st.rerun()
                with b2:
                    if job['status'] == "Completed" and not (job.get('calc_volume') and job.get('calc_sentiment')):
                        if st.button("✨ Add AI", key=f"retro_{job['id']}", help="Retroactively add Volume & Sentiment"):
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
                        st.rerun()"""

code = code.replace(ui_old, ui_new)

# Now update the Worker to handle retroactive jobs
worker_old = """            update_status("Loading audio file...", 10)
            audio = whisperx.load_audio(audio_path)
            
            update_status(f"Loading Whisper model ({whisper_model})...", 15)"""

worker_new = """            
            retroactive_only = job_data.get("retroactive_only", False)
            if retroactive_only:
                update_status("Calculating acoustic and emotional tone...", 10)
                try:
                    import pydub
                    from pydub import AudioSegment
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    
                    full_audio = AudioSegment.from_file(audio_path)
                    analyzer = SentimentIntensityAnalyzer()
                    
                    with open(result_file, "r") as f: res_data = json.load(f)
                    segments = res_data.get("segments", [])
                    total_segs = len(segments)
                    
                    for i, seg in enumerate(segments):
                        start_sec = seg.get("start", 0)
                        end_sec = seg.get("end", 0)
                        text = seg.get("text", "").strip()
                        
                        # Volume & Speech Rate
                        start_ms = int(start_sec * 1000)
                        end_ms = int(end_sec * 1000)
                        chunk = full_audio[start_ms:end_ms]
                        dbfs = chunk.dBFS
                        if dbfs == float('-inf'): dbfs = -100.0
                        
                        duration_min = (end_sec - start_sec) / 60.0
                        word_count = len(text.split())
                        wpm = word_count / duration_min if duration_min > 0 else 0
                        
                        seg["dbfs"] = round(dbfs, 1)
                        seg["wpm"] = round(wpm, 1)
                        
                        # Sentiment
                        vs = analyzer.polarity_scores(text)
                        compound = vs['compound']
                        if compound >= 0.05: sentiment = "Positive"
                        elif compound <= -0.05: sentiment = "Negative"
                        else: sentiment = "Neutral"
                        
                        seg["sentiment"] = sentiment
                        seg["sentiment_score"] = compound
                        
                        if i % max(1, int(total_segs/20)) == 0:
                            update_status("Adding AI Metrics...", 10 + int(85 * (i/max(1, total_segs))))
                            
                    with open(result_file, "w") as f: json.dump(res_data, f, ensure_ascii=False)
                    update_status("Completed", 100)
                except Exception as e:
                    update_status("Failed", 0, error=f"Retro error: {e}")
                continue
                
            update_status("Loading audio file...", 10)
            audio = whisperx.load_audio(audio_path)
            
            update_status(f"Loading Whisper model ({whisper_model})...", 15)"""

code = code.replace(worker_old, worker_new)

with open('/home/gingerbin/.openclaw/workspace/projects/whisper_standalone/whisper_app.py', 'w') as f:
    f.write(code)
    print("Done")
