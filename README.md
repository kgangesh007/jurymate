# ⚖️ JuryMate — AI-Powered Hackathon Evaluation Assistant

## Project Structure
```
jurymate/
├── app.py                  # Streamlit UI (main entry point)
├── requirements.txt        # Python dependencies
├── core/
│   ├── database.py         # SQLite — all 4 tables
│   └── ingestion.py        # Parse → Chunk → Embed → ChromaDB
├── agents/
│   └── jury_agents.py      # Scoring, Chat, Comparison, Feedback agents
├── data/                   # Auto-created: SQLite DB + ChromaDB
└── uploads/                # Temp upload storage
```

## Day 1 Setup

### Step 1 — AMD Developer Cloud
```bash
# SSH into your AMD instance, then:
git clone <your-repo>
cd jurymate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Install and start Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3        # or: ollama pull deepseek-r1
```

### Step 4 — Run the app
```bash
streamlit run app.py --server.port 8501
```

Open: http://localhost:8501

---

## Architecture

```
PDF/PPT Upload
      │
      ▼
SQLite documents table (status: pending → processing → indexed/failed)
      │
      ▼
PyMuPDF / python-pptx  →  LangChain TextSplitter
      │
      ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
      │
      ▼
ChromaDB — Single collection, metadata: {team, doc_id, filename, page}
      │
      ▼
Hybrid Retriever: Vector (60%) + BM25 (40%)
      │
      ▼
LangGraph Agents: Scoring | Chat | Comparison | Feedback
      │
      ▼
Llama 3 / Deepseek (via Ollama on AMD GPU)
      │
      ▼
Streamlit UI — ChatGPT-style sidebar per team
```

## Database Schema
- **documents** — registry with status tracking
- **teams** — registered teams
- **scores** — criteria-wise scores (10/40/20/15/15)
- **chat_history** — per-team conversation history

## MVP Phases
- **MVP 1** (Day 1-2): Upload + RAG + Scoring ✅
- **MVP 2** (Day 3-4): Jury chat + Comparison + Leaderboard ✅
- **MVP 3** (Future): Video analysis, Code analysis, Cross-validation
