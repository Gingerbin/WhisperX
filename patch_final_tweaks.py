import sys
import re

with open('/home/gingerbin/.openclaw/workspace/projects/whisper_standalone/whisper_app.py', 'r') as f:
    code = f.read()

# 1. Add model to the caption in the job list
code = code.replace(
    '''st.caption(f"ID: {job['id']} | Tools: {tools_str}")''',
    '''st.caption(f"ID: {job['id']} | Model: {job.get('model', 'base')} | Tools: {tools_str}")'''
)

# 2. Update CSS for sticky player and remove old play-btn css
css_target = """    .flag-green { background-color: #27ae60 !important; color: white; }
    /* Force action buttons to sit side-by-side even on mobile */
    div.stButton { display: inline-block !important; width: auto !important; margin-right: 0.5rem !important; }
    div.stButton > button { width: auto !important; }
    
    /* Play button styling */
    .play-btn {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        font-size: 16px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        float: left;
        margin-top: 5px;
    }
    .play-btn:hover { background-color: #0056b3; }"""

new_css = """    .flag-green { background-color: #27ae60 !important; color: white; }
    /* Force action buttons to sit side-by-side even on mobile */
    div.stButton { display: inline-block !important; width: auto !important; margin-right: 0.5rem !important; }
    div.stButton > button { width: auto !important; }
    
    /* Sticky Audio Player */
    [data-testid="stAudio"] {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        z-index: 99999;
        background-color: #1e1e1e; /* dark mode safe */
        padding: 10px 20px;
        box-shadow: 0px -5px 15px rgba(0,0,0,0.5);
        border-top: 1px solid #333;
    }
    .main .block-container {
        padding-bottom: 120px !important;
    }
    
    /* Inline JS Play Button */
    .inline-play {
        cursor: pointer;
        background-color: #007bff;
        color: white;
        border-radius: 50%;
        width: 22px;
        height: 22px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        margin-left: 8px;
        vertical-align: middle;
        text-decoration: none;
    }
    .inline-play:hover { background-color: #0056b3; }"""

code = code.replace(css_target, new_css)

# 3. Add global audio player and remove sticky state header
old_audio = """        # Audio Player (Sticky Header basically)
        if st.session_state.play_audio_start is not None:
            st.success(f"🎧 Playing snippet from {format_time(st.session_state.play_audio_start)}")
            st.audio(job_meta.get("filepath"), start_time=int(st.session_state.play_audio_start), autoplay=True)
            if st.button("⏹️ Clear Audio"):
                st.session_state.play_audio_start = None
                st.rerun()"""

new_audio = """        # Global Sticky Audio Player
        if os.path.exists(job_meta.get("filepath", "")):
            st.audio(job_meta.get("filepath"))"""
code = code.replace(old_audio, new_audio)

# 4. Modify the tab 0 logic (Transcript) to include sliders and new play button layout
old_tab0 = """        with tabs[0]:
            for i, seg in enumerate(segments):
                spk_id = seg.get("speaker", "UNKNOWN")
                spk_name = st.session_state.speaker_map.get(spk_id, spk_id)
                color = get_speaker_color(spk_id)
                start_time = format_time(seg.get('start', 0))
                end_time = format_time(seg.get('end', 0))
                
                m_parts = []
                if "dbfs" in seg: 
                    m_parts.append(f"🔊 {seg['dbfs']}dB (Gain: {seg.get('gain_adj',0):+g}dB)")
                    dr = seg.get('dynamic_range',0)
                    m_parts.append(f"📈 DR: <span class='{'text-red' if dr > 25 else ''}'>{dr}dB</span>")
                    crest = seg.get('crest_factor',0)
                    m_parts.append(f"🏔️ Crest: <span class='{'text-red' if crest > 20 else ''}'>{crest}dB</span>")
                    m_parts.append(f"🔇 Sil: {int(seg.get('silence_density',0)*100)}%")
                    wpm = seg.get('wpm', 0)
                    m_parts.append(f"⏱️ <span class='{'text-red' if wpm > 220 else ''}'>{wpm} wpm</span>")
                    
                if "sentiment" in seg: 
                    sent = seg['sentiment']
                    m_parts.append(f"🎭 <span class='{'text-red' if sent == 'Negative' else ''}'>{sent}</span>")
                
                metrics_html = f"<span class='metrics-badge'>{' | '.join(m_parts)}</span>" if m_parts else ""
                
                flags_html = ""
                for f in seg.get("audio_flags", []):
                    flags_html += f"<span class='audio-flag'>{f}</span>"
                for f in seg.get("psych_flags", []):
                    extra_class = ""
                    if f in ["You Talk", "Criticism", "Contempt", "Defensiveness", "Stonewalling", "Gaslighting", "Victimhood", "Dark Triad"]: extra_class = "flag-red"
                    elif f in ["We Talk", "Restorative Positive"]: extra_class = "flag-green"
                    flags_html += f"<span class='psych-flag {extra_class}'>{f}</span>"
                
                card_html = f\"\"\"<div class="segment-card" style="border-left-color: {color}; display: block; overflow: hidden;">
                    <span class="speaker-badge" style="background-color: {color}; float:left;">{spk_name}</span>
                    <span class="timestamp" style="float:left; margin-top:0.25em;">[{start_time} - {end_time}]</span>
                    {metrics_html}
                    <div style='clear:both;'></div>
                    <div style='margin-top:0.5rem; font-size:1.1em;'>{seg.get('text', '')} {flags_html}</div>
                </div>\"\"\"
                
                col_btn, col_card = st.columns([1, 15])
                with col_btn:
                    if st.button("▶️", key=f"play_{i}_{job_id}"):
                        st.session_state.play_audio_start = seg.get("start", 0)
                        st.rerun()
                with col_card:
                    st.markdown(card_html, unsafe_allow_html=True)"""

new_tab0 = """        with tabs[0]:
            col_t1, col_t2 = st.columns(2)
            with col_t1: dr_thresh = st.number_input("🔴 High Dynamic Range (DR) Threshold", value=25, step=1, help="If DR exceeds this, it turns red.")
            with col_t2: crest_thresh = st.number_input("🔴 High Crest Factor Threshold", value=20, step=1, help="If Crest Factor exceeds this, it turns red.")
            st.markdown("---")
            
            for i, seg in enumerate(segments):
                spk_id = seg.get("speaker", "UNKNOWN")
                spk_name = st.session_state.speaker_map.get(spk_id, spk_id)
                color = get_speaker_color(spk_id)
                start_sec = seg.get('start', 0)
                start_time = format_time(start_sec)
                end_time = format_time(seg.get('end', 0))
                
                m_parts = []
                if "dbfs" in seg: 
                    m_parts.append(f"🔊 {seg['dbfs']}dB (Gain: {seg.get('gain_adj',0):+g}dB)")
                    dr = seg.get('dynamic_range',0)
                    m_parts.append(f"📈 DR: <span class='{'text-red' if dr > dr_thresh else ''}'>{dr}dB</span>")
                    crest = seg.get('crest_factor',0)
                    m_parts.append(f"🏔️ Crest: <span class='{'text-red' if crest > crest_thresh else ''}'>{crest}dB</span>")
                    m_parts.append(f"🔇 Sil: {int(seg.get('silence_density',0)*100)}%")
                    wpm = seg.get('wpm', 0)
                    m_parts.append(f"⏱️ <span class='{'text-red' if wpm > 220 else ''}'>{wpm} wpm</span>")
                    
                if "sentiment" in seg: 
                    sent = seg['sentiment']
                    m_parts.append(f"🎭 <span class='{'text-red' if sent == 'Negative' else ''}'>{sent}</span>")
                
                metrics_html = f"<span class='metrics-badge'>{' | '.join(m_parts)}</span>" if m_parts else ""
                
                flags_html = ""
                for f in seg.get("audio_flags", []):
                    flags_html += f"<span class='audio-flag'>{f}</span>"
                for f in seg.get("psych_flags", []):
                    extra_class = ""
                    if f in ["You Talk", "Criticism", "Contempt", "Defensiveness", "Stonewalling", "Gaslighting", "Victimhood", "Dark Triad"]: extra_class = "flag-red"
                    elif f in ["We Talk", "Restorative Positive"]: extra_class = "flag-green"
                    flags_html += f"<span class='psych-flag {extra_class}'>{f}</span>"
                
                play_btn = f"<a class='inline-play' onclick=\"const a=document.querySelector('audio'); if(a){{a.currentTime={start_sec}; a.play();}}\" title='Play snippet'>▶</a>"
                
                card_html = f\"\"\"<div class="segment-card" style="border-left-color: {color}; display: block; overflow: hidden;">
                    <span class="speaker-badge" style="background-color: {color}; float:left;">{spk_name}</span>
                    <span class="timestamp" style="float:left; margin-top:0.25em;">[{start_time} - {end_time}]</span>
                    {play_btn}
                    {metrics_html}
                    <div style='clear:both;'></div>
                    <div style='margin-top:0.5rem; font-size:1.1em;'>{seg.get('text', '')} {flags_html}</div>
                </div>\"\"\"
                
                st.markdown(card_html, unsafe_allow_html=True)"""

code = code.replace(old_tab0, new_tab0)

# 5. Modify Vader summary in Psych stats
old_tab1_start = """                for s in segments:
                    dur = s.get("end", 0) - s.get("start", 0)
                    speaker_times[s.get("speaker", "UNKNOWN")] = speaker_times.get(s.get("speaker", "UNKNOWN"), 0) + dur
                    total_time += dur
                
                for spk, stats in psych_stats.items():"""

new_tab1_start = """                vader_stats = {}
                for s in segments:
                    dur = s.get("end", 0) - s.get("start", 0)
                    spk = s.get("speaker", "UNKNOWN")
                    speaker_times[spk] = speaker_times.get(spk, 0) + dur
                    total_time += dur
                    
                    if spk not in vader_stats: vader_stats[spk] = {"pos": 0, "neg": 0, "neu": 0, "total": 0}
                    sent = s.get("sentiment")
                    if sent == "Positive": vader_stats[spk]["pos"] += 1
                    elif sent == "Negative": vader_stats[spk]["neg"] += 1
                    elif sent == "Neutral": vader_stats[spk]["neu"] += 1
                    vader_stats[spk]["total"] += 1
                
                for spk, stats in psych_stats.items():"""

code = code.replace(old_tab1_start, new_tab1_start)

# Add the sentiment to the expander
old_expander = """                        st.markdown(f"**🎭 Roles Detected:** `{'` `'.join(stats.get('roles', ['Neutral']))}`")
                        col_a, col_b, col_c = st.columns(3)"""

new_expander = """                        st.markdown(f"**🎭 Roles Detected:** `{'` `'.join(stats.get('roles', ['Neutral']))}`")
                        v_stat = vader_stats.get(spk, {"pos":0, "neg":0, "neu":0, "total":0})
                        v_tot = max(v_stat["total"], 1)
                        st.markdown(f"**🧠 VADER Sentiment Overall:** {v_stat['pos']} Pos ({v_stat['pos']/v_tot*100:.0f}%) | {v_stat['neg']} Neg ({v_stat['neg']/v_tot*100:.0f}%) | {v_stat['neu']} Neu ({v_stat['neu']/v_tot*100:.0f}%)")
                        st.markdown("---")
                        col_a, col_b, col_c = st.columns(3)"""
                        
code = code.replace(old_expander, new_expander)

# Finally, write the file
with open('/home/gingerbin/.openclaw/workspace/projects/whisper_standalone/whisper_app.py', 'w') as f:
    f.write(code)

