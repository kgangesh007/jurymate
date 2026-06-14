import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

from core.database import (
    init_db, get_all_teams, get_documents, get_chat_history,
    save_message, get_score, get_all_scores, save_score,
    clear_chat_history, delete_team
)
from core.ingestion import ingest_document, ingest_code, delete_team_documents
from agents.jury_agents import (
    score_submission, score_all_teams, chat_with_team,
    chat_with_all, compare_teams, generate_feedback, cross_validate
)

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="JuryMate",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; color: #e8e8f0; }
[data-testid="stSidebar"] { background: #16181f; border-right: 1px solid #2a2d3a; }
.jury-logo {
    font-size: 1.4rem; font-weight: 700; letter-spacing: -0.5px;
    color: #fff; padding: 0.5rem 0 1.2rem 0;
    border-bottom: 1px solid #2a2d3a; margin-bottom: 1rem;
}
.jury-logo span { color: #7c6af5; }
.chat-user {
    background: #7c6af5; color: #fff; padding: 0.75rem 1rem;
    border-radius: 18px 18px 4px 18px; margin: 0.4rem 0 0.4rem 20%;
    font-size: 0.9rem; line-height: 1.5;
}
.chat-assistant {
    background: #1e2030; color: #e8e8f0; padding: 0.75rem 1rem;
    border-radius: 18px 18px 18px 4px; margin: 0.4rem 20% 0.4rem 0;
    font-size: 0.9rem; line-height: 1.6; border: 1px solid #2a2d3a;
}
.upload-hint {
    text-align: center; color: #9a9bb0; font-size: 0.85rem; padding: 1rem;
    border: 1px dashed #2a2d3a; border-radius: 10px; margin-bottom: 1rem;
}
.metric-tile {
    background: #1e2030; border: 1px solid #2a2d3a;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #7c6af5; }
.metric-label { font-size: 0.78rem; color: #9a9bb0; margin-top: 2px; }
.score-sub {
    background: #12141c; border-radius: 8px; padding: 0.5rem 0.8rem;
    margin: 0.3rem 0; font-size: 0.82rem;
}
.score-sub-label { color: #9a9bb0; }
.score-sub-val   { color: #7c6af5; font-weight: 700; }
.validate-verified   { color: #4ade80; }
.validate-unverified { color: #f87171; }
.validate-bonus      { color: #fbbf24; }
.section-label {
    font-size: 0.75rem; color: #9a9bb0; padding: 0.5rem 0 0.3rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.stButton > button {
    background: #7c6af5; color: #fff; border: none;
    border-radius: 8px; font-weight: 600; padding: 0.4rem 1rem;
}
.stButton > button:hover { background: #6a58e0; }
div[data-testid="stChatInput"] > div {
    border: 1px solid #2a2d3a; background: #1e2030; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────
if "active_view"    not in st.session_state:
    st.session_state.active_view    = "all_teams"
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

# ── Chat renderer ──────────────────────────────────────────
def render_chat(chat_key: str, is_all_teams: bool = False,
                team_name: str = None):
    history = get_chat_history(chat_key)

    if not history:
        hint = (
            "Ask anything across all submissions — who used what tech, "
            "how they built it, compare approaches, ask about code."
            if is_all_teams else
            "Ask about this team's presentation OR code — "
            "what functions do, how components work, what tech they used."
        )
        st.markdown(f'<div class="upload-hint">{hint}</div>',
                    unsafe_allow_html=True)

    for msg in history:
        cls = "chat-user" if msg["role"] == "user" else "chat-assistant"
        st.markdown(f'<div class="{cls}">{msg["message"]}</div>',
                    unsafe_allow_html=True)

    question = st.chat_input(
        "Ask about all teams…" if is_all_teams else "Ask about presentation or code…"
    )

    if question:
        save_message(chat_key, "user", question)
        with st.spinner("Thinking…"):
            result = (
                chat_with_all(question, history)
                if is_all_teams else
                chat_with_team(team_name, question, history)
            )
        answer = result["answer"]
        if result.get("citations"):
            answer += "\n\n---\n**Sources:**\n" + \
                      "\n".join(f"📌 {c}" for c in result["citations"])
        save_message(chat_key, "assistant", answer)
        st.rerun()

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="jury-logo">⚖️ Jury<span>Mate</span></div>',
                unsafe_allow_html=True)

    # Upload submission
    with st.expander("➕ Add Submission", expanded=False):
        team_input  = st.text_input("Team name", placeholder="e.g. Team Alpha")
        upload_type = st.radio("Upload type", ["📄 PPT / PDF", "💻 Code (ZIP or file)"],
                               horizontal=True)
        accepted    = (
            ["pdf", "pptx", "ppt"]
            if "PPT" in upload_type else
            ["zip", "py", "js", "ts", "java", "go", "rs", "cpp", "c"]
        )
        uploaded = st.file_uploader(
            "Upload file", type=accepted, label_visibility="collapsed"
        )
        if st.button("Upload & Index", use_container_width=True):
            if team_input and uploaded:
                with st.spinner(f"Indexing {uploaded.name}..."):
                    tname = team_input.strip().lower().replace(" ", "_")
                    if "PPT" in upload_type:
                        result = ingest_document(tname, uploaded.name, uploaded.read())
                    else:
                        result = ingest_code(tname, uploaded.name, uploaded.read())

                if result["success"]:
                    chunks = result.get("total_chunks", 0)
                    files  = result.get("total_files", result.get("total_pages", 0))
                    st.success(f"✅ {chunks} chunks indexed ({files} files)")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")
            else:
                st.warning("Enter team name and upload a file")

    st.markdown("---")

    # All Teams Chat
    is_all = st.session_state.active_view == "all_teams"
    if st.button(f"{'●' if is_all else '○'}  💬 All Teams Chat",
                 use_container_width=True, key="nav_all"):
        st.session_state.active_view = "all_teams"
        st.rerun()

    # Team list
    teams = get_all_teams()
    if teams:
        st.markdown('<div class="section-label">Submissions</div>',
                    unsafe_allow_html=True)
        for t in teams:
            name      = t["team_name"]
            is_active = st.session_state.active_view == name
            if st.button(
                f"{'●' if is_active else '○'}  📁 {name.replace('_',' ').title()}",
                use_container_width=True, key=f"nav_{name}"
            ):
                st.session_state.active_view = name
                st.rerun()

    st.markdown("---")

    # Tools
    st.markdown('<div class="section-label">Tools</div>',
                unsafe_allow_html=True)
    for label, key, icon in [
        ("Leaderboard",   "leaderboard", "🏆"),
        ("Compare Teams", "compare",     "⚖️"),
        ("Registry",      "registry",    "📋"),
    ]:
        is_active = st.session_state.active_view == key
        if st.button(f"{'●' if is_active else '○'}  {icon} {label}",
                     use_container_width=True, key=f"nav_{key}"):
            st.session_state.active_view = key
            st.rerun()

# ══════════════════════════════════════════════════════════
# VIEWS
# ══════════════════════════════════════════════════════════
view = st.session_state.active_view

# ── VIEW: All Teams Chat ───────────────────────────────────
if view == "all_teams":
    st.markdown("## 💬 All Teams Chat")
    st.caption("Ask about any team's presentation OR code across all submissions")
    render_chat("__all_teams__", is_all_teams=True)

# ── VIEW: Individual Team ──────────────────────────────────
elif view in [t["team_name"] for t in get_all_teams()]:
    team_name    = view
    display_name = team_name.replace("_", " ").title()
    docs         = get_documents(team_name)
    existing_score = get_score(team_name)

    # Check what's uploaded
    has_ppt  = any(d["file_type"] in ("pdf","pptx") and d["status"]=="indexed" for d in docs)
    has_code = any(d["file_type"] == "code" and d["status"]=="indexed" for d in docs)

    # Header
    col_title, col_score, col_validate, col_feedback, col_clear, col_del = \
        st.columns([2.5, 1.3, 1.3, 1.2, 0.7, 0.7])

    with col_title:
        st.markdown(f"## 📁 {display_name}")
        badges = []
        if has_ppt:  badges.append("📄 PPT")
        if has_code: badges.append("💻 Code")
        if badges:   st.caption(" · ".join(badges))

    with col_score:
        if st.button("⭐ Score", use_container_width=True):
            with st.spinner("Scoring…"):
                result = score_submission(team_name)
            if "error" not in result:
                scores = {
                    "problem":      result["problem"]["score"],
                    "technical":    result["technical"]["score"],
                    "future_work":  result["future_work"]["score"],
                    "innovation":   result["innovation"]["score"],
                    "presentation": result["presentation"]["score"],
                    "total":        result["total"]
                }
                reasoning = {
                    k: result[k].get("reason","")
                    for k in ["problem","technical","future_work",
                              "innovation","presentation"]
                }
                save_score(team_name, scores, reasoning)
                st.session_state[f"last_score_{team_name}"] = result
                st.rerun()
            else:
                st.error(result["error"])

    with col_validate:
        if has_ppt and has_code:
            if st.button("🔍 Validate", use_container_width=True):
                with st.spinner("Cross-validating PPT vs Code…"):
                    val_result = cross_validate(team_name)
                st.session_state[f"validate_{team_name}"] = val_result
                st.rerun()
        else:
            st.caption("Upload both PPT + Code to validate")

    with col_feedback:
        if existing_score:
            if st.button("📝 Feedback", use_container_width=True):
                with st.spinner("Generating feedback…"):
                    fb = generate_feedback(team_name, existing_score)
                st.session_state[f"feedback_{team_name}"] = fb
                st.rerun()

    with col_clear:
        if st.button("🗑️", use_container_width=True):
            clear_chat_history(team_name)
            st.rerun()

    with col_del:
        if st.session_state.confirm_delete == team_name:
            if st.button("✅", use_container_width=True):
                delete_team_documents(team_name)
                delete_team(team_name)
                st.session_state.active_view    = "all_teams"
                st.session_state.confirm_delete = None
                st.rerun()
        else:
            if st.button("❌", use_container_width=True):
                st.session_state.confirm_delete = team_name
                st.rerun()

    if st.session_state.confirm_delete == team_name:
        st.warning(f"⚠️ Delete **{display_name}** and all data? Click ✅ to confirm.")

    # Score breakdown
    last_score = st.session_state.get(f"last_score_{team_name}")
    if existing_score:
        with st.expander("📊 Score Breakdown", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            tiles = [
                (c1, "Total",        existing_score["total"],        100),
                (c2, "Technical",    existing_score["technical"],     40),
                (c3, "Future Work",  existing_score["future_work"],   20),
                (c4, "Innovation",   existing_score["innovation"],    15),
                (c5, "Presentation", existing_score["presentation"],  15),
                (c6, "Problem",      existing_score["problem"],       10),
            ]
            for col, label, val, max_val in tiles:
                with col:
                    st.markdown(
                        f'<div class="metric-tile">'
                        f'<div class="metric-value">{int(val)}</div>'
                        f'<div class="metric-label">{label}<br>/{max_val}</div>'
                        f'</div>', unsafe_allow_html=True
                    )

            # Sub-criteria from last score
            if last_score and isinstance(last_score.get("technical"), dict):
                st.markdown("#### Sub-criteria Detail")
                tech = last_score["technical"]
                prob = last_score["problem"]
                inno = last_score["innovation"]
                pres = last_score["presentation"]
                fw   = last_score["future_work"]

                col_l, col_r = st.columns(2)
                with col_l:
                    for label, val, max_v in [
                        ("AI/ML Approach",      tech.get("ai_approach",0),      10),
                        ("Working Solution",     tech.get("working_solution",0), 10),
                        ("Code Quality",         tech.get("code_quality",0),     10),
                        ("Framework Usage",      tech.get("framework_usage",0),  10),
                        ("Problem Clarity",      prob.get("clarity",0),           5),
                        ("Problem Relevance",    prob.get("relevance",0),         5),
                    ]:
                        st.markdown(
                            f'<div class="score-sub">'
                            f'<span class="score-sub-label">{label}: </span>'
                            f'<span class="score-sub-val">{val}/{max_v}</span>'
                            f'</div>', unsafe_allow_html=True
                        )
                with col_r:
                    for label, val, max_v in [
                        ("Novelty",             inno.get("novelty",0),          8),
                        ("Unique Approach",     inno.get("unique_approach",0),  7),
                        ("Scalability",         fw.get("scalability",0),        10),
                        ("Impact Measurement",  fw.get("impact",0),             10),
                        ("Explanation Clarity", pres.get("clarity",0),           8),
                        ("Demo Quality",        pres.get("demo_quality",0),      7),
                    ]:
                        st.markdown(
                            f'<div class="score-sub">'
                            f'<span class="score-sub-label">{label}: </span>'
                            f'<span class="score-sub-val">{val}/{max_v}</span>'
                            f'</div>', unsafe_allow_html=True
                        )

    # Cross validation result
    val_result = st.session_state.get(f"validate_{team_name}")
    if val_result:
        with st.expander("🔍 PPT vs Code Validation", expanded=True):
            if "error" in val_result:
                st.error(val_result["error"])
            else:
                st.markdown(val_result["answer"])
                if val_result.get("citations"):
                    st.caption("Sources: " + " | ".join(val_result["citations"]))

    # Feedback
    feedback = st.session_state.get(f"feedback_{team_name}")
    if feedback:
        with st.expander("📝 Auto Feedback", expanded=True):
            st.markdown(feedback)

    # Doc status
    if docs:
        status_map = {"indexed":"🟢","processing":"🟡","pending":"🔵","failed":"🔴"}
        pills = " &nbsp; ".join(
            f"{status_map.get(d['status'],'⚪')} {d['filename']}"
            for d in docs
        )
        st.caption(pills, unsafe_allow_html=True)

    st.markdown("---")
    render_chat(team_name, team_name=team_name)

# ── VIEW: Leaderboard ──────────────────────────────────────
elif view == "leaderboard":
    st.markdown("## 🏆 Leaderboard")

    # Score All button
    teams = get_all_teams()
    if teams:
        col_btn, col_info = st.columns([2, 6])
        with col_btn:
            if st.button("⭐ Score All Teams", use_container_width=True):
                names = [t["team_name"] for t in teams]
                progress = st.progress(0, text="Scoring teams…")
                for i, name in enumerate(names):
                    progress.progress((i+1)/len(names), text=f"Scoring {name}…")
                    result = score_submission(name)
                    if "error" not in result:
                        scores = {
                            "problem":      result["problem"]["score"],
                            "technical":    result["technical"]["score"],
                            "future_work":  result["future_work"]["score"],
                            "innovation":   result["innovation"]["score"],
                            "presentation": result["presentation"]["score"],
                            "total":        result["total"]
                        }
                        reasoning = {k: result[k].get("reason","")
                                     for k in ["problem","technical","future_work",
                                               "innovation","presentation"]}
                        save_score(name, scores, reasoning)
                progress.empty()
                st.rerun()

    scores = get_all_scores()
    if not scores:
        st.info("No scores yet. Score individual teams or click Score All Teams.")
    else:
        df = pd.DataFrame(scores)
        df["team_name"] = df["team_name"].str.replace("_"," ").str.title()
        df = df.sort_values("total", ascending=False).reset_index(drop=True)
        df.index += 1

        fig = px.bar(
            df, x="team_name", y="total", color="total",
            color_continuous_scale=["#2a2d3a","#7c6af5"],
            labels={"team_name":"Team","total":"Total Score"}, text="total"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8f0", showlegend=False,
            coloraxis_showscale=False, margin=dict(t=20,b=20)
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        if len(scores) >= 2:
            criteria = ["problem","technical","future_work","innovation","presentation"]
            fig2 = go.Figure()
            for _, row in df.iterrows():
                orig_name = row["team_name"].lower().replace(" ","_")
                s = next((sc for sc in scores if sc["team_name"] == orig_name), None)
                if s:
                    fig2.add_trace(go.Scatterpolar(
                        r=[s[c] for c in criteria] + [s[criteria[0]]],
                        theta=criteria + [criteria[0]],
                        fill="toself", name=row["team_name"]
                    ))
            fig2.update_layout(
                polar=dict(bgcolor="#1e2030",
                           radialaxis=dict(visible=True, color="#9a9bb0")),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0",
                margin=dict(t=30,b=30)
            )
            st.plotly_chart(fig2, use_container_width=True)

        display_cols = {
            "team_name":   "Team",     "total":       "Total /100",
            "technical":   "Tech /40", "future_work": "Future /20",
            "innovation":  "Innov /15","presentation":"Pres /15",
            "problem":     "Prob /10"
        }
        st.dataframe(
            df[list(display_cols.keys())].rename(columns=display_cols),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 💬 Ask About the Leaderboard")
    render_chat("__leaderboard_chat__", is_all_teams=True)

# ── VIEW: Compare ──────────────────────────────────────────
elif view == "compare":
    st.markdown("## ⚖️ Compare Teams")
    teams = get_all_teams()
    names = [t["team_name"] for t in teams]

    if len(names) < 2:
        st.info("Upload at least 2 team submissions to compare.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            team_a = st.selectbox("Team A", names,
                                  format_func=lambda x: x.replace("_"," ").title(),
                                  key="cmp_a")
        with c2:
            team_b = st.selectbox("Team B",
                                  [n for n in names if n != team_a],
                                  format_func=lambda x: x.replace("_"," ").title(),
                                  key="cmp_b")

        if st.button("⚖️ Run Comparison"):
            with st.spinner("Comparing…"):
                result = compare_teams(team_a, team_b)
            st.markdown("### Analysis")
            st.markdown(result["answer"])
            if result.get("citations"):
                with st.expander("📌 Sources"):
                    for c in result["citations"]:
                        st.caption(c)

        score_a = get_score(team_a)
        score_b = get_score(team_b)
        if score_a and score_b:
            st.markdown("### Score Comparison")
            criteria = ["problem","technical","future_work","innovation","presentation"]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=team_a.replace("_"," ").title(),
                x=criteria, y=[score_a[c] for c in criteria],
                marker_color="#7c6af5"
            ))
            fig.add_trace(go.Bar(
                name=team_b.replace("_"," ").title(),
                x=criteria, y=[score_b[c] for c in criteria],
                marker_color="#4ade80"
            ))
            fig.update_layout(
                barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0",
                margin=dict(t=20,b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💬 Ask About These Teams")
    render_chat("__compare_chat__", is_all_teams=True)

# ── VIEW: Registry ─────────────────────────────────────────
elif view == "registry":
    st.markdown("## 📋 Document Registry")
    docs = get_documents()

    if not docs:
        st.info("No documents uploaded yet.")
    else:
        status_emoji = {
            "indexed":    "✅ Indexed",
            "processing": "⏳ Processing",
            "pending":    "🔵 Pending",
            "failed":     "❌ Failed"
        }
        rows = []
        for d in docs:
            rows.append({
                "Team":      d["team_name"].replace("_"," ").title(),
                "File":      d["filename"],
                "Type":      d["file_type"].upper() if d["file_type"] else "-",
                "Size (KB)": d["file_size_kb"] or "-",
                "Pages/Files": d["total_pages"] or "-",
                "Chunks":    d["total_chunks"] or "-",
                "Status":    status_emoji.get(d["status"], d["status"]),
                "Uploaded":  d["upload_time"][:16] if d["upload_time"] else "-",
                "Error":     d["error_message"] or ""
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        for col, val, label in [
            (m1, len(docs),                                          "Total Files"),
            (m2, sum(1 for d in docs if d["status"]=="indexed"),     "Indexed"),
            (m3, sum(1 for d in docs if d["status"]=="failed"),      "Failed"),
            (m4, sum(d["total_chunks"] or 0 for d in docs),          "Total Chunks"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>', unsafe_allow_html=True
                )