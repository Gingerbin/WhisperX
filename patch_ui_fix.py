import sys
import re

with open('/home/gingerbin/.openclaw/workspace/projects/whisper_standalone/whisper_app.py', 'r') as f:
    code = f.read()

# 1. Fix the double-rendering HTML issue
# Streamlit st.markdown() sometimes sanitizes nested HTML if not wrapped correctly.
# The issue is {metrics_html} and {flags_html} being printed as text rather than rendering as HTML.
# We will combine them into the single card_html string securely.

old_card_gen = """                card_html = f\"\"\"<div class="segment-card" data-start="{start_sec}" data-end="{end_sec}" data-color="{color}" style="border-left-color: {color};">
                    <div style="margin-bottom: 8px; display: flex; align-items: center;">
                        <a class='inline-play' data-time='{start_sec}' title='Play from here'>▶</a>
                        <span class="speaker-badge" style="background-color: {color}; margin-bottom: 0;">{spk_name}</span>
                        <span class="timestamp">[{start_time} - {end_time}]</span>
                        <a class='toggle-btn' data-target='stats-{i}' title='Toggle Stats'>📊</a>
                    </div>
                    <div style='font-size:1.1em; margin-bottom:0.5rem;'>{seg.get('text', '')}</div>
                    
                    <div id='stats-{i}' style='display:none; font-size: 0.85em; color:#666; background: rgba(0,0,0,0.05); padding: 5px; border-radius: 5px; margin-bottom: 5px;'>
                        {metrics_html}
                    </div>
                    
                    <div>{flags_html}</div>
                </div>\"\"\"
                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Editor Expander
                with st.expander(f"⚙️ Edit Segment / Add Note"):
                    c_spk, c_note, c_act = st.columns([2, 5, 2])
                    with c_spk:
                        new_spk = st.selectbox("Speaker", options=speakers, index=speakers.index(spk_id) if spk_id in speakers else 0, key=f"edit_spk_{i}_{job_id}")
                    with c_note:
                        new_text = st.text_area("Add note / transcript block after:", key=f"edit_txt_{i}_{job_id}")
                        is_transcription = st.checkbox("Analyze as spoken words", key=f"is_trans_{i}_{job_id}")
                    with c_act:
                        if st.button("💾 Save", key=f"save_seg_{i}_{job_id}", use_container_width=True):
                            changed = False
                            if new_spk != spk_id:
                                segments[i]["speaker"] = new_spk
                                changed = True
                            if new_text.strip():
                                new_seg = {
                                    "start": end_sec, "end": end_sec + 2.0,
                                    "speaker": new_spk,
                                    "text": new_text.strip() if is_transcription else f"[NOTE: {new_text.strip()}]"
                                }
                                if is_transcription:
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
                                                    
                                segments.insert(i+1, new_seg)
                                changed = True
                                
                            if changed:
                                res["segments"] = segments
                                with open(result_file, "w") as f: json.dump(res, f, ensure_ascii=False)
                                st.rerun()"""


new_card_gen = """                
                is_editing = st.session_state.get(f"editing_{i}_{job_id}", False)
                is_adding = st.session_state.get(f"adding_{i}_{job_id}", False)
                
                edit_btn_html = f"<a class='action-icon' onclick=\\\"window.parent.document.getElementById('hidden_edit_{i}_{job_id}').click();\\\" title='Edit Speaker'>⚙️</a>"
                add_btn_html = f"<a class='action-icon' onclick=\\\"window.parent.document.getElementById('hidden_add_{i}_{job_id}').click();\\\" title='Add Note/Block'>➕</a>"
                
                card_html = f\"\"\"<div class="segment-card" data-start="{start_sec}" data-end="{end_sec}" data-color="{color}" style="border-left-color: {color};">
                    <div style="margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center;">
                            <a class='inline-play' data-time='{start_sec}' title='Play from here'>▶</a>
                            <span class="speaker-badge" style="background-color: {color}; margin-bottom: 0;">{spk_name}</span>
                            <span class="timestamp">[{start_time} - {end_time}]</span>
                            <a class='toggle-btn' data-target='stats-{i}' title='Toggle Stats'>📊</a>
                        </div>
                        <div style="display: flex; gap: 5px;">
                            {edit_btn_html}
                            {add_btn_html}
                        </div>
                    </div>
                    <div style='font-size:1.1em; margin-bottom:0.5rem;'>{seg.get('text', '')}</div>
                    
                    <div id='stats-{i}' style='display:none; font-size: 0.85em; color:#666; background: rgba(0,0,0,0.05); padding: 5px; border-radius: 5px; margin-bottom: 5px;'>
                        {metrics_html}
                    </div>
                    
                    <div>{flags_html}</div>
                </div>\"\"\"
                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Hidden buttons to trigger streamlit state from custom JS HTML links
                # We put them in an empty container so they don't break layout
                with st.container():
                    st.markdown(f"<div style='display:none;'>", unsafe_allow_html=True)
                    if st.button("edit", key=f"hidden_edit_{i}_{job_id}"):
                        st.session_state[f"editing_{i}_{job_id}"] = not is_editing
                        st.session_state[f"adding_{i}_{job_id}"] = False
                        st.rerun()
                    if st.button("add", key=f"hidden_add_{i}_{job_id}"):
                        st.session_state[f"adding_{i}_{job_id}"] = not is_adding
                        st.session_state[f"editing_{i}_{job_id}"] = False
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                if is_editing:
                    st.markdown(f"**⚙️ Edit Speaker for Block {i+1}**")
                    col_e1, col_e2 = st.columns([3, 1])
                    with col_e1:
                        new_spk = st.selectbox("Assign to:", options=speakers, index=speakers.index(spk_id) if spk_id in speakers else 0, key=f"edit_spk_{i}_{job_id}")
                    with col_e2:
                        if st.button("Save Speaker", key=f"save_edit_{i}_{job_id}", use_container_width=True):
                            segments[i]["speaker"] = new_spk
                            res["segments"] = segments
                            with open(result_file, "w") as f: json.dump(res, f, ensure_ascii=False)
                            st.session_state[f"editing_{i}_{job_id}"] = False
                            st.rerun()
                            
                if is_adding:
                    st.markdown(f"**➕ Add Block Beneath ({format_time(end_sec + 0.001)})**")
                    is_transcription = st.checkbox("This is spoken dialogue (Analyze for psych stats)", key=f"is_trans_{i}_{job_id}")
                    
                    if is_transcription:
                        col_a1, col_a2, col_a3 = st.columns([1, 1, 2])
                        with col_a1: new_end_sec = st.number_input("End Time (seconds)", value=end_sec + 2.0, min_value=end_sec, step=0.5, key=f"end_sec_{i}_{job_id}")
                        with col_a2: note_spk = st.selectbox("Speaker", options=speakers, key=f"add_spk_{i}_{job_id}")
                    else:
                        new_end_sec = end_sec + 0.1
                        note_spk = spk_id
                        
                    # Custom text area without cmd+enter restriction by just using standard streamlit widget
                    new_text = st.text_area("Content:", key=f"edit_txt_{i}_{job_id}", help="Just type and click Save below.")
                    
                    if st.button("💾 Save New Block", key=f"save_add_{i}_{job_id}"):
                        if new_text.strip():
                            new_seg = {
                                "start": end_sec + 0.001,
                                "end": new_end_sec,
                                "speaker": note_spk,
                                "text": new_text.strip() if is_transcription else f"[NOTE: {new_text.strip()}]"
                            }
                            if is_transcription:
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
                                                
                            segments.insert(i+1, new_seg)
                            res["segments"] = segments
                            with open(result_file, "w") as f: json.dump(res, f, ensure_ascii=False)
                            st.session_state[f"adding_{i}_{job_id}"] = False
                            st.rerun()"""

code = code.replace(old_card_gen, new_card_gen)

# Need to inject CSS for action-icon
css_target = """    .toggle-btn:hover { background-color: #e67e22; }"""
new_css = """    .toggle-btn:hover { background-color: #e67e22; }
    
    .action-icon {
        cursor: pointer; background-color: #f1f2f6; border-radius: 4px;
        padding: 4px; font-size: 14px; margin-left: 5px; vertical-align: middle; user-select: none; text-decoration: none !important;
        border: 1px solid #ddd; display: inline-block;
    }
    .action-icon:hover { background-color: #dfe4ea; }"""
    
code = code.replace(css_target, new_css)

with open('/home/gingerbin/.openclaw/workspace/projects/whisper_standalone/whisper_app.py', 'w') as f:
    f.write(code)

