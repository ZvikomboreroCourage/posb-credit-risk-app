from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "posb_credit_app.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            role TEXT,
            action TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()

    existing = cur.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    if existing == 0:
        cur.execute(
            "INSERT INTO users(username, password_hash, role, full_name) VALUES(?,?,?,?)",
            ("admin", hash_password("admin123"), "Admin", "System Admin"),
        )
        conn.commit()
    conn.close()


def create_user(username: str, password: str, role: str, full_name: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO users(username, password_hash, role, full_name) VALUES(?,?,?,?)",
        (username.strip(), hash_password(password), role, full_name.strip()),
    )
    conn.commit()
    conn.close()


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT username, password_hash, role, full_name FROM users WHERE username = ?",
        (username.strip(),),
    ).fetchone()
    conn.close()
    if row and row["password_hash"] == hash_password(password):
        return dict(row)
    return None


def log_action(username: str | None, role: str | None, action: str, details: str = "") -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO audit_log(username, role, action, details) VALUES(?,?,?,?)",
        (username, role, action, details[:4000]),
    )
    conn.commit()
    conn.close()


def read_recent_logs(limit: int = 100):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows
