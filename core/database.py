import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/jurymate.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    c = conn.cursor()

    # Teams table
    c.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name  TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Document registry
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name     TEXT NOT NULL,
            filename      TEXT NOT NULL,
            file_type     TEXT,
            file_size_kb  INTEGER,
            total_pages   INTEGER,
            total_chunks  INTEGER,
            upload_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            indexed_time  TIMESTAMP,
            status        TEXT DEFAULT 'pending',
            error_message TEXT
        )
    """)

    # Scores table
    c.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name    TEXT NOT NULL,
            problem      FLOAT,
            technical    FLOAT,
            future_work  FLOAT,
            innovation   FLOAT,
            presentation FLOAT,
            total        FLOAT,
            reasoning    TEXT,
            scored_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Chat history
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name  TEXT NOT NULL,
            role       TEXT NOT NULL,
            message    TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

# ── Teams ──────────────────────────────────────────────────
def add_team(team_name: str):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO teams (team_name) VALUES (?)",
            (team_name,)
        )
        conn.commit()
    finally:
        conn.close()

def get_all_teams():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM teams ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── Documents ──────────────────────────────────────────────
def register_document(team_name: str, filename: str,
                       file_type: str, file_size_kb: int) -> int:
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO documents (team_name, filename, file_type,
                               file_size_kb, status)
        VALUES (?, ?, ?, ?, 'pending')
    """, (team_name, filename, file_type, file_size_kb))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def update_document_status(doc_id: int, status: str,
                            total_pages: int = None,
                            total_chunks: int = None,
                            error_message: str = None):
    conn = get_connection()
    indexed_time = datetime.now().isoformat() if status == "indexed" else None
    conn.execute("""
        UPDATE documents
        SET status        = ?,
            total_pages   = COALESCE(?, total_pages),
            total_chunks  = COALESCE(?, total_chunks),
            indexed_time  = COALESCE(?, indexed_time),
            error_message = COALESCE(?, error_message)
        WHERE id = ?
    """, (status, total_pages, total_chunks,
          indexed_time, error_message, doc_id))
    conn.commit()
    conn.close()

def get_documents(team_name: str = None):
    conn = get_connection()
    if team_name:
        rows = conn.execute(
            "SELECT * FROM documents WHERE team_name = ? ORDER BY upload_time DESC",
            (team_name,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY upload_time DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── Scores ─────────────────────────────────────────────────
def save_score(team_name: str, scores: dict, reasoning: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO scores
            (team_name, problem, technical, future_work,
             innovation, presentation, total, reasoning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        team_name,
        scores.get("problem", 0),
        scores.get("technical", 0),
        scores.get("future_work", 0),
        scores.get("innovation", 0),
        scores.get("presentation", 0),
        scores.get("total", 0),
        str(reasoning)
    ))
    conn.commit()
    conn.close()

def get_score(team_name: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM scores WHERE team_name = ? ORDER BY scored_at DESC LIMIT 1",
        (team_name,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_scores():
    conn = get_connection()
    rows = conn.execute("""
        SELECT s.*
        FROM scores s
        INNER JOIN (
            SELECT team_name, MAX(scored_at) as latest
            FROM scores GROUP BY team_name
        ) latest ON s.team_name = latest.team_name
                    AND s.scored_at = latest.latest
        ORDER BY s.total DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── Chat History ───────────────────────────────────────────
def save_message(team_name: str, role: str, message: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO chat_history (team_name, role, message)
        VALUES (?, ?, ?)
    """, (team_name, role, message))
    conn.commit()
    conn.close()

def get_chat_history(team_name: str):
    conn = get_connection()
    rows = conn.execute("""
        SELECT role, message FROM chat_history
        WHERE team_name = ?
        ORDER BY created_at
    """, (team_name,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def clear_chat_history(team_name: str):
    conn = get_connection()
    conn.execute(
        "DELETE FROM chat_history WHERE team_name = ?",
        (team_name,)
    )
    conn.commit()
    conn.close()

# ── Delete Team ────────────────────────────────────────────
def delete_team(team_name: str):
    """Delete team and all related data from SQLite"""
    conn = get_connection()
    conn.execute("DELETE FROM chat_history WHERE team_name = ?", (team_name,))
    conn.execute("DELETE FROM scores WHERE team_name = ?", (team_name,))
    conn.execute("DELETE FROM documents WHERE team_name = ?", (team_name,))
    conn.execute("DELETE FROM teams WHERE team_name = ?", (team_name,))
    conn.commit()
    conn.close()