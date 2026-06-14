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
    """4-layer JSON extraction"""
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```",     "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
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

# ── Scoring Agent — Sub-criteria based ─────────────────────
SCORE_SYSTEM = """
You are an expert hackathon judge. Score the submission using EXACT sub-criteria below.
Be consistent — same quality should always get same score.

SCORING RUBRIC (total 100):

1. PROBLEM DEFINITION (max 10):
   - Clarity of problem statement: 0-5
   - Real-world/enterprise relevance: 0-5

2. TECHNICAL IMPLEMENTATION (max 40):
   - Correct AI/ML approach: 0-10
   - Working solution evidence: 0-10
   - Code quality and structure: 0-10
   - Framework usage effectiveness: 0-10

3. FUTURE WORK (max 20):
   - Scalability and deployment plan: 0-10
   - Impact measurement and roadmap: 0-10

4. INNOVATION (max 15):
   - Novelty of idea: 0-8
   - Unique implementation approach: 0-7

5. PRESENTATION (max 15):
   - Clarity of explanation: 0-8
   - Demo quality and flow: 0-7

Return ONLY valid JSON. No text before or after. No markdown. No backticks.
Exact format:
{"problem":{"clarity":4,"relevance":4,"score":8,"reason":"one sentence"},"technical":{"ai_approach":8,"working_solution":8,"code_quality":7,"framework_usage":8,"score":31,"reason":"one sentence"},"future_work":{"scalability":8,"impact":7,"score":15,"reason":"one sentence"},"innovation":{"novelty":6,"unique_approach":5,"score":11,"reason":"one sentence"},"presentation":{"clarity":6,"demo_quality":5,"score":11,"reason":"one sentence"},"total":76,"summary":"2-3 sentence overall summary"}
"""

def score_submission(team_name: str) -> dict:
    # Get both PPT and code context
    ppt_chunks  = retrieve_context(
        "problem solution technology innovation impact architecture",
        team_name=team_name,
        n_results=5,
        content_type="presentation"
    )
    code_chunks = retrieve_context(
        "implementation functions classes imports technology stack",
        team_name=team_name,
        n_results=4,
        content_type="code"
    )

    all_chunks = ppt_chunks + code_chunks
    if not all_chunks:
        # Fallback — search without content_type filter
        all_chunks = retrieve_context(
            "problem solution technology innovation",
            team_name=team_name,
            n_results=6
        )

    if not all_chunks:
        return {"error": "No content found. Make sure submission is uploaded and indexed."}

    ppt_context  = "\n\n".join(c["text"] for c in ppt_chunks) if ppt_chunks else "No presentation found"
    code_context = "\n\n".join(c["text"] for c in code_chunks) if code_chunks else "No code found"
    citations    = _build_citations(all_chunks)

    user_prompt = f"""
PRESENTATION CONTENT:
{ppt_context}

CODE CONTENT:
{code_context}

Score this hackathon submission using the exact rubric provided.
"""
    raw    = _llm(SCORE_SYSTEM, user_prompt)
    print("\n===== RAW SCORING OUTPUT =====")
    print(raw)
    print("==============================\n")

    parsed = _parse_json(raw)
    if parsed:
        for key in ["problem","technical","future_work","innovation","presentation"]:
            if key not in parsed:
                parsed[key] = {"score": 0, "reason": "Not evaluated"}
            elif isinstance(parsed[key], (int, float)):
                parsed[key] = {"score": int(parsed[key]), "reason": ""}

        parsed["total"] = sum(
            parsed[k]["score"]
            for k in ["problem","technical","future_work","innovation","presentation"]
            if isinstance(parsed[k], dict)
        )
        if "summary" not in parsed:
            parsed["summary"] = "Evaluation complete."
        parsed["citations"] = citations
        return parsed

    return {"error": "Could not parse scores. Check terminal.", "raw": raw}

# ── Cross Validation Agent ──────────────────────────────────
CROSS_VALIDATE_SYSTEM = """
You are a technical auditor verifying if a team's code matches their presentation claims.

Check each claim from the presentation against the actual code.
Be specific about what was found and what was missing.

Return a structured analysis:
1. VERIFIED CLAIMS - things mentioned in PPT that exist in code
2. UNVERIFIED CLAIMS - things mentioned in PPT but NOT found in code
3. BONUS FINDINGS - good things in code not mentioned in PPT
4. CONSISTENCY SCORE - overall score out of 10
5. VERDICT - one paragraph summary

Format your response in clear sections with these exact headers.
"""

def cross_validate(team_name: str) -> dict:
    """Check if PPT claims match actual code"""
    ppt_chunks = retrieve_context(
        "technology framework architecture approach claims built",
        team_name=team_name,
        n_results=6,
        content_type="presentation"
    )
    code_chunks = retrieve_context(
        "import library function class implementation",
        team_name=team_name,
        n_results=6,
        content_type="code"
    )

    if not ppt_chunks:
        return {"error": "No presentation found for this team."}
    if not code_chunks:
        return {"error": "No code found for this team. Upload code first."}

    ppt_context  = "\n\n".join(c["text"] for c in ppt_chunks)
    code_context = "\n\n".join(c["text"] for c in code_chunks)
    citations    = _build_citations(ppt_chunks + code_chunks)

    user_prompt = f"""
PRESENTATION CLAIMS:
{ppt_context}

ACTUAL CODE:
{code_context}

Verify if the code matches the presentation claims.
"""
    answer = _llm(CROSS_VALIDATE_SYSTEM, user_prompt)
    return {"answer": answer, "citations": citations}

# ── Chat Agent ──────────────────────────────────────────────
CHAT_SYSTEM_TEAM = """
You are a jury assistant with access to ONE team's hackathon submission.
You have access to both their PRESENTATION and their CODE.
Answer questions clearly based on the retrieved context.
When answering about code:
- Explain what files do
- Explain what functions do
- Explain how components work together
- Identify technologies and frameworks used
Always cite your source (filename, slide/page) at the end.
If you cannot find the answer say so honestly.
"""

CHAT_SYSTEM_ALL = """
You are a jury assistant with access to ALL team submissions including both presentations and code.
When comparing teams be specific about which team did what.
You can answer questions about:
- Which teams used which technologies
- How different teams approached the same problem
- Code quality comparisons
- Architecture differences
Always cite sources at the end. Be objective and fair.
"""

def chat_with_team(team_name: str, question: str,
                   history: list[dict]) -> dict:
    # Search both presentation and code
    ppt_chunks  = retrieve_context(question, team_name=team_name,
                                   n_results=3, content_type="presentation")
    code_chunks = retrieve_context(question, team_name=team_name,
                                   n_results=3, content_type="code")
    all_chunks  = retrieve_context(question, team_name=team_name, n_results=5)

    # Use specific chunks if found, else fallback
    chunks    = (ppt_chunks + code_chunks) if (ppt_chunks or code_chunks) else all_chunks
    citations = _build_citations(chunks)

    ppt_ctx  = "\n\n".join(c["text"] for c in ppt_chunks)  if ppt_chunks  else ""
    code_ctx = "\n\n".join(c["text"] for c in code_chunks) if code_chunks else ""
    all_ctx  = "\n\n".join(c["text"] for c in all_chunks)  if not (ppt_chunks or code_chunks) else ""

    context = ""
    if ppt_ctx:
        context += f"PRESENTATION:\n{ppt_ctx}\n\n"
    if code_ctx:
        context += f"CODE:\n{code_ctx}\n\n"
    if all_ctx:
        context += all_ctx

    user_prompt = f"""
Previous conversation:
{_format_history(history)}

Context:
{context}

Question: {question}
"""
    answer = _llm(CHAT_SYSTEM_TEAM, user_prompt)
    return {"answer": answer, "citations": citations}

def chat_with_all(question: str, history: list[dict]) -> dict:
    chunks    = retrieve_context(question, n_results=10)
    context   = "\n\n".join(
        f"[{c['team'].upper()} | {c.get('content_type','').upper()}] {c['text']}"
        for c in chunks
    )
    citations = _build_citations(chunks)

    user_prompt = f"""
Previous conversation:
{_format_history(history)}

Context from all teams:
{context}

Question: {question}
"""
    answer = _llm(CHAT_SYSTEM_ALL, user_prompt)
    return {"answer": answer, "citations": citations}

# ── Comparison Agent ────────────────────────────────────────
COMPARE_SYSTEM = """
You are a hackathon judge comparing two team submissions.
You have access to both their presentations AND code.
Compare across these criteria:
1. Problem Definition and Relevance
2. Technical Implementation (including code quality)
3. Innovation and Creativity
4. Future Work and Scalability
5. Presentation Quality

For each criterion state which team did better and why with specific evidence.
End with an Overall Verdict.
"""

def compare_teams(team_a: str, team_b: str) -> dict:
    chunks_a = retrieve_context(
        "problem solution technology innovation code",
        team_name=team_a, n_results=5
    )
    chunks_b = retrieve_context(
        "problem solution technology innovation code",
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

Compare these two submissions across all criteria.
"""
    answer = _llm(COMPARE_SYSTEM, user_prompt)
    return {"answer": answer, "citations": citations}

# ── Feedback Agent ──────────────────────────────────────────
FEEDBACK_SYSTEM = """
You are a senior hackathon mentor writing constructive feedback.
Be encouraging but honest and specific.
Structure:
- Strengths (2-3 points with specific examples from their work)
- Areas for Improvement (2-3 points with actionable suggestions)
- Advice for Next Steps (1-2 points)
Keep under 250 words. Reference actual content from their submission.
"""

def generate_feedback(team_name: str, scores: dict) -> str:
    chunks  = retrieve_context(
        "solution approach technology results impact",
        team_name=team_name, n_results=5
    )
    context = "\n\n".join(c["text"] for c in chunks)
    scores_text = json.dumps({
        k: v for k, v in scores.items()
        if k in ["problem","technical","future_work","innovation","presentation","total"]
    }, indent=2)

    return _llm(FEEDBACK_SYSTEM, f"""
Team: {team_name}
Scores: {scores_text}
Submission content: {context}
Write constructive feedback for this team.
""")

# ── Score All Teams ─────────────────────────────────────────
def score_all_teams(team_names: list[str]) -> dict:
    """Score multiple teams, return dict of results"""
    results = {}
    for team in team_names:
        print(f"\n🔄 Scoring {team}...")
        results[team] = score_submission(team)
    return results

# ── Helpers ─────────────────────────────────────────────────
def _build_citations(chunks: list[dict]) -> list[str]:
    seen, citations = set(), []
    for c in chunks:
        ct    = c.get("content_type", "")
        label = (
            f"[CODE] {c['filename']}"
            if ct == "code" else
            f"[PPT] {c['filename']} → Slide/Page {c['page']}"
        )
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