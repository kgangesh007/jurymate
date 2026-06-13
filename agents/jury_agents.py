import json
import re
import ollama
from core.ingestion import retrieve_context

# Change to "llama3" on AMD Developer Cloud
MODEL = "deepseek-r1:70b"

# ── LLM helper ─────────────────────────────────────────────
def _llm(system: str, user: str) -> str:
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
    )
    return resp["message"]["content"]

def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response — handles all 3b/8b model quirks"""
    # Layer 1 — strip markdown fences
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```",     "", text)
    text = text.strip()

    # Layer 2 — direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Layer 3 — find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # Layer 4 — fix trailing commas + single quotes
    try:
        cleaned = re.sub(r",\s*}", "}", text)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        cleaned = cleaned.replace("'", '"')
        match2  = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match2:
            return json.loads(match2.group())
    except Exception:
        pass

    return {}

# ── Scoring Agent ───────────────────────────────────────────
SCORE_SYSTEM = """
You are an expert hackathon judge evaluating AI project submissions.
Score the submission using these exact criteria:

1. problem      - Problem Definition and Relevance        (max 10)
2. technical    - Technical Implementation                (max 40)
3. future_work  - Learnings and Future Work               (max 20)
4. innovation   - Innovation and Creativity               (max 15)
5. presentation - Presentation and Demo Quality           (max 15)

Return ONLY a valid JSON object. No explanation. No markdown. No backticks.
Start with { and end with }.

Exact format:
{"problem":{"score":8,"reason":"one sentence"},"technical":{"score":30,"reason":"one sentence"},"future_work":{"score":15,"reason":"one sentence"},"innovation":{"score":12,"reason":"one sentence"},"presentation":{"score":12,"reason":"one sentence"},"total":77,"summary":"2-3 sentence overall summary"}
"""

def score_submission(team_name: str) -> dict:
    chunks = retrieve_context(
        "problem solution technology innovation impact architecture",
        team_name=team_name,
        n_results=6
    )
    if not chunks:
        return {
            "error": "No content found. Make sure the submission was uploaded and indexed successfully."
        }

    context   = "\n\n".join(c["text"] for c in chunks)
    citations = _build_citations(chunks)

    raw = _llm(SCORE_SYSTEM, f"Score this hackathon submission:\n{context}")

    # Debug — shows in terminal
    print("\n========== RAW LLM SCORING OUTPUT ==========")
    print(raw)
    print("=============================================\n")

    parsed = _parse_json(raw)

    if parsed:
        # Ensure all keys exist with safe defaults
        for key in ["problem", "technical", "future_work",
                    "innovation", "presentation"]:
            if key not in parsed:
                parsed[key] = {"score": 0, "reason": "Not evaluated"}
            elif isinstance(parsed[key], (int, float)):
                parsed[key] = {"score": int(parsed[key]), "reason": ""}

        # Recalculate total safely
        parsed["total"] = sum(
            parsed[k]["score"]
            for k in ["problem", "technical", "future_work",
                      "innovation", "presentation"]
            if isinstance(parsed[k], dict)
        )

        if "summary" not in parsed:
            parsed["summary"] = "Evaluation complete."

        parsed["citations"] = citations
        return parsed

    return {
        "error": "Could not parse scores. Check terminal for raw output.",
        "raw":   raw
    }

# ── Chat Agent ──────────────────────────────────────────────
CHAT_SYSTEM_TEAM = """
You are a jury assistant with access to ONE team's hackathon submission.
The context may include text extracted from slides as well as visual
descriptions of diagrams and images from those slides.
Answer questions clearly based only on the retrieved context.
Always cite your source (filename, slide or page number) at the end.
If you cannot find the answer say so honestly — do not make things up.
"""

CHAT_SYSTEM_ALL = """
You are a jury assistant with access to ALL team submissions.
The context may include text and visual descriptions from slides.
When comparing or ranking teams be specific about which team did what.
Always cite sources (team name, filename, slide or page) at the end.
Be objective and fair across all teams.
"""

def chat_with_team(team_name: str, question: str,
                   history: list[dict]) -> dict:
    chunks    = retrieve_context(question, team_name=team_name, n_results=5)
    context   = "\n\n".join(c["text"] for c in chunks)
    citations = _build_citations(chunks)

    history_text = _format_history(history)
    user_prompt  = f"""
Previous conversation:
{history_text}

Retrieved context from submission:
{context}

Current question: {question}
"""
    answer = _llm(CHAT_SYSTEM_TEAM, user_prompt)
    return {"answer": answer, "citations": citations}

def chat_with_all(question: str, history: list[dict]) -> dict:
    chunks    = retrieve_context(question, n_results=8)
    context   = "\n\n".join(
        f"[{c['team'].upper()}] {c['text']}" for c in chunks
    )
    citations = _build_citations(chunks)

    history_text = _format_history(history)
    user_prompt  = f"""
Previous conversation:
{history_text}

Retrieved context from all teams:
{context}

Current question: {question}
"""
    answer = _llm(CHAT_SYSTEM_ALL, user_prompt)
    return {"answer": answer, "citations": citations}

# ── Comparison Agent ────────────────────────────────────────
COMPARE_SYSTEM = """
You are a hackathon judge comparing two team submissions side by side.
The context may include visual descriptions of architecture diagrams.
For each criterion state which team did better and why:
1. Problem Definition and Relevance
2. Technical Implementation
3. Innovation and Creativity
4. Future Work and Scalability
5. Presentation Quality

End with an Overall Verdict recommending which submission is stronger.
Be specific and reference actual content from both submissions.
"""

def compare_teams(team_a: str, team_b: str) -> dict:
    chunks_a = retrieve_context(
        "problem solution technology innovation architecture",
        team_name=team_a, n_results=5
    )
    chunks_b = retrieve_context(
        "problem solution technology innovation architecture",
        team_name=team_b, n_results=5
    )

    context_a = "\n".join(c["text"] for c in chunks_a)
    context_b = "\n".join(c["text"] for c in chunks_b)
    citations = _build_citations(chunks_a + chunks_b)

    user_prompt = f"""
Team A ({team_a}):
{context_a}

Team B ({team_b}):
{context_b}

Compare these two submissions across all evaluation criteria.
"""
    answer = _llm(COMPARE_SYSTEM, user_prompt)
    return {"answer": answer, "citations": citations}

# ── Feedback Agent ──────────────────────────────────────────
FEEDBACK_SYSTEM = """
You are a senior hackathon mentor writing constructive feedback for a team.
Be encouraging but honest and specific.
Structure your feedback as:
- Strengths (2-3 points)
- Areas for Improvement (2-3 points)
- Advice for Next Steps (1-2 points)
Keep total length under 200 words.
"""

def generate_feedback(team_name: str, scores: dict) -> str:
    chunks  = retrieve_context(
        "solution approach technology results impact",
        team_name=team_name, n_results=4
    )
    context = "\n\n".join(c["text"] for c in chunks)
    scores_text = json.dumps({
        k: v for k, v in scores.items()
        if k in ["problem", "technical", "future_work",
                 "innovation", "presentation", "total"]
    }, indent=2)

    user_prompt = f"""
Team: {team_name}
Scores: {scores_text}
Submission content: {context}

Write constructive feedback for this team.
"""
    return _llm(FEEDBACK_SYSTEM, user_prompt)

# ── Helpers ─────────────────────────────────────────────────
def _build_citations(chunks: list[dict]) -> list[str]:
    seen      = set()
    citations = []
    for c in chunks:
        label = f"{c['filename']} -> Slide/Page {c['page']}"
        if label not in seen:
            seen.add(label)
            citations.append(label)
    return citations

def _format_history(history: list[dict]) -> str:
    if not history:
        return "None"
    lines = []
    for msg in history[-6:]:
        role = "Jury" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['message']}")
    return "\n".join(lines)