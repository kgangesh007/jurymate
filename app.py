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
from core.ingestion import ingest_document, delete_team_documents
from agents.jury_agents import (
    score_submission, chat_with_team,
    chat_with_all, compare_teams, generate_feedback
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
    color: #ffffff; padding: 0.5rem 0 1.2rem 0;
    border-bottom: 1px solid #2a2d3a; margin-bottom: 1rem;
}
.jury-logo span { color: #7c6af5; }
.chat-user {
    background: #7c6af5; color: #fff;
    padding: 0.75rem 1rem; border-radius: 18px 18px 4px 18px;
    margin: 0.4rem 0 0.4rem 20%; font-size: 0.9rem; line-height: 1.5;
}
.chat-assistant {
    background: #1e2030; color: #e8e8f0;
    padding: 0.75rem 1rem; border-radius: 18px 18px 18px 4px;
    margin: 0.4rem 20% 0.4rem 0; font-size: 0.9rem;
    line-height: 1.6; border: 1px solid #2a2d3a;
}
.upload-hint {
    text-align: center; color: #9a9bb0; font-size: 0.85rem;
    padding: 1rem; border: 1px dashed #2a2d3a;
    border-radius: 10px; margin-bottom: 1rem;
}
.metric-tile {
    background: #1e2030; border: 1px solid #2a2d3a;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #7c6af5; }
.metric-label { font-size: 0.78rem; color: #9a9bb0; margin-top: 2px; }
.stButton > button {
    background: #7c6af5; color: #fff; border: none;
    border-radius: 8px; font-weight: 600; padding: 0.4rem 1rem;
}
.stButton > button:hover { background: #6a58e0; }
.delete-btn > button {
    background: #3a1a1a !important; color: #f87171 !important;
}
div[data-testid="stChatInput"] > div {
    border: 1px solid #2a2d3a; background: #1e2030; border-radius: 12px;
}
.section-label {
    font-size: 0.75rem; color: #9a9bb0; padding: 0.5rem 0 0.3rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────
if "active_view" not in st.session_state:
    st.session_state.active_view = "all_teams"
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

# ── Helpers ────────────────────────────────────────────────
def render_chat(chat_key: str, is_all_teams: bool = False,
                team_name: str = None):
    history = get_chat_history(chat_key)

    if not history:
        st.markdown(
            '<div class="upload-hint">'
            + ("Ask anything across all team submissions — who used what, how they built it, which is best."
               if is_all_teams else
               f"Ask anything about this submission.")
            + "</div>",
            unsafe_allow_html=True
        )
    for msg in history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{msg["message"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-assistant">{msg["message"]}</div>',
                unsafe_allow_html=True
            )

    placeholder = ("Ask about all teams — who used RAG? which team had best innovation?…"
                   if is_all_teams else
                   f"Ask about this submission…")
    question = st.chat_input(placeholder)

    if question:
        save_message(chat_key, "user", question)
        with st.spinner("Thinking…"):
            if is_all_teams:
                result = chat_with_all(question, history)
            else:
                result = chat_with_team(team_name, question, history)

        answer = result["answer"]
        if result.get("citations"):
            cite_text = "\n".join(f"📌 {c}" for c in result["citations"])
            answer += f"\n\n---\n**Sources:**\n{cite_text}"

        save_message(chat_key, "assistant", answer)
        st.rerun()

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="jury-logo">⚖️ Jury<span>Mate</span></div>',
                unsafe_allow_html=True)

    # Upload
    with st.expander("➕ Add Submission", expanded=False):
        team_input = st.text_input("Team name", placeholder="e.g. Team Alpha")
        uploaded   = st.file_uploader(
            "Upload PDF or PPT",
            type=["pdf", "pptx", "ppt"],
            label_visibility="collapsed"
        )
        if st.button("Upload & Index", use_container_width=True):
            if team_input and uploaded:
                with st.spinner(f"Indexing {uploaded.name}..."):
                    result = ingest_document(
                        team_name=team_input.strip().lower().replace(" ", "_"),
                        filename=uploaded.name,
                        file_bytes=uploaded.read()
                    )
                if result["success"]:
                    st.success(f"✅ {result['total_chunks']} chunks indexed")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")
            else:
                st.warning("Enter team name and upload a file")

    st.markdown("---")

    # All Teams Chat
    is_all = st.session_state.active_view == "all_teams"
    if st.button(
        f"{'●' if is_all else '○'}  💬 All Teams Chat",
        use_container_width=True, key="nav_all"
    ):
        st.session_state.active_view = "all_teams"
        st.rerun()

    # Per-team list
    teams = get_all_teams()
    if teams:
        st.markdown('<div class="section-label">Submissions</div>',
                    unsafe_allow_html=True)
        for t in teams:
            name      = t["team_name"]
            is_active = st.session_state.active_view == name
            label     = f"{'●' if is_active else '○'}  📁 {name.replace('_',' ').title()}"
            if st.button(label, use_container_width=True, key=f"nav_{name}"):
                st.session_state.active_view = name
                st.rerun()

    st.markdown("---")

    # Tools
    st.markdown('<div class="section-label">Tools</div>',
                unsafe_allow_html=True)
    for label, view_key, icon in [
        ("Leaderboard",   "leaderboard", "🏆"),
        ("Compare Teams", "compare",     "⚖️"),
        ("Registry",      "registry",    "📋"),
    ]:
        is_active = st.session_state.active_view == view_key
        if st.button(
            f"{'●' if is_active else '○'}  {icon} {label}",
            use_container_width=True, key=f"nav_{view_key}"
        ):
            st.session_state.active_view = view_key
            st.rerun()

# ══════════════════════════════════════════════════════════
# VIEWS
# ══════════════════════════════════════════════════════════
view = st.session_state.active_view

# ── VIEW: All Teams Chat ───────────────────────────────────
if view == "all_teams":
    st.markdown("## 💬 All Teams Chat")
    st.caption("Chat across every submission — ask who used what technology, compare approaches, find the best solutions")
    render_chat("__all_teams__", is_all_teams=True)

# ── VIEW: Individual Team ──────────────────────────────────
elif view in [t["team_name"] for t in get_all_teams()]:
    team_name    = view
    display_name = team_name.replace("_", " ").title()
    docs         = get_documents(team_name)
    existing_score = get_score(team_name)

    # Header row
    col_title, col_score, col_clear, col_delete = st.columns([3.5, 1.5, 0.8, 0.8])
    with col_title:
        st.markdown(f"## 📁 {display_name}")
    with col_score:
        if st.button("⭐ Score This Team", use_container_width=True):
            with st.spinner("Scoring submission…"):
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
                    k: result[k].get("reason", "")
                    for k in ["problem","technical","future_work",
                              "innovation","presentation"]
                }
                save_score(team_name, scores, reasoning)
                st.rerun()
            else:
                st.error(result["error"])
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            clear_chat_history(team_name)
            st.rerun()
    with col_delete:
        if st.session_state.confirm_delete == team_name:
            if st.button("✅ Confirm", use_container_width=True):
                delete_team_documents(team_name)
                delete_team(team_name)
                st.session_state.active_view = "all_teams"
                st.session_state.confirm_delete = None
                st.rerun()
        else:
            if st.button("❌ Delete", use_container_width=True):
                st.session_state.confirm_delete = team_name
                st.rerun()

    # Confirm delete warning
    if st.session_state.confirm_delete == team_name:
        st.warning(f"⚠️ Are you sure you want to delete **{display_name}** and all their data? Click Confirm to proceed.")

    # Score breakdown
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
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # Doc status
    if docs:
        status_map = {"indexed":"🟢","processing":"🟡","pending":"🔵","failed":"🔴"}
        pills = " &nbsp; ".join(
            f"{status_map.get(d['status'],'⚪')} {d['filename']}" for d in docs
        )
        st.caption(pills, unsafe_allow_html=True)

    st.markdown("---")
    render_chat(team_name, team_name=team_name)

# ── VIEW: Leaderboard ──────────────────────────────────────
elif view == "leaderboard":
    st.markdown("## 🏆 Leaderboard")

    scores = get_all_scores()
    if not scores:
        st.info("No scores yet. Open a team and click ⭐ Score This Team.")
    else:
        df = pd.DataFrame(scores)
        df["team_name"] = df["team_name"].str.replace("_"," ").str.title()
        df = df.sort_values("total", ascending=False).reset_index(drop=True)
        df.index += 1

        fig = px.bar(
            df, x="team_name", y="total",
            color="total",
            color_continuous_scale=["#2a2d3a","#7c6af5"],
            labels={"team_name":"Team","total":"Total Score"},
            text="total"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8f0", showlegend=False,
            coloraxis_showscale=False, margin=dict(t=20,b=20)
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        display_cols = {
            "team_name":   "Team",
            "total":       "Total /100",
            "technical":   "Technical /40",
            "future_work": "Future /20",
            "innovation":  "Innovation /15",
            "presentation":"Presentation /15",
            "problem":     "Problem /10"
        }
        st.dataframe(
            df[list(display_cols.keys())].rename(columns=display_cols),
            use_container_width=True
        )

    # Chat below leaderboard
    st.markdown("---")
    st.markdown("### 💬 Ask About the Leaderboard")
    st.caption("Ask questions like: Which team scored highest on innovation? Who had best technical implementation?")
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
            team_a = st.selectbox(
                "Team A", names,
                format_func=lambda x: x.replace("_"," ").title(),
                key="cmp_a"
            )
        with c2:
            team_b = st.selectbox(
                "Team B",
                [n for n in names if n != team_a],
                format_func=lambda x: x.replace("_"," ").title(),
                key="cmp_b"
            )

        if st.button("⚖️ Run Comparison", use_container_width=False):
            with st.spinner("Comparing submissions…"):
                result = compare_teams(team_a, team_b)
            st.markdown("### Analysis")
            st.markdown(result["answer"])
            if result.get("citations"):
                with st.expander("📌 Sources"):
                    for c in result["citations"]:
                        st.caption(c)

        # Score comparison chart
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
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8e8f0",
                margin=dict(t=20,b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Chat below compare
    st.markdown("---")
    st.markdown("### 💬 Ask About These Teams")
    st.caption("Ask questions across all submissions — who used better technology? which team solved the bigger problem?")
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
                "Pages":     d["total_pages"] or "-",
                "Chunks":    d["total_chunks"] or "-",
                "Status":    status_emoji.get(d["status"], d["status"]),
                "Uploaded":  d["upload_time"][:16] if d["upload_time"] else "-",
                "Error":     d["error_message"] or ""
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        total_docs   = len(docs)
        indexed_docs = sum(1 for d in docs if d["status"] == "indexed")
        failed_docs  = sum(1 for d in docs if d["status"] == "failed")
        total_chunks = sum(d["total_chunks"] or 0 for d in docs)

        for col, val, label in [
            (m1, total_docs,   "Total Files"),
            (m2, indexed_docs, "Indexed"),
            (m3, failed_docs,  "Failed"),
            (m4, total_chunks, "Total Chunks"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )