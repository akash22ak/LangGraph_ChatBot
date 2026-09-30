import time
import sqlite3
import threading
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

DB_PATH = "ChatBot.DB"


# ── Database connections ───────────────────────────────────────
# Two separate connections: one owned by the LangGraph checkpointer, one for our own
# tables. WAL mode + a busy timeout lets them coexist without "database is locked" errors.

def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


checkpointer_conn = _connect()
checkpointer = SqliteSaver(conn=checkpointer_conn)

app_conn = _connect()
_app_lock = threading.RLock()   # serialises access to app_conn across Streamlit threads


# ── LangGraph ──────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ── Titles table ───────────────────────────────────────────────

def setup_tables():
    with _app_lock:
        app_conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_titles (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        app_conn.commit()


# ── Title / thread management ──────────────────────────────────

def save_thread_title(thread_id, title: str):
    """Insert a title, or update it if the thread already has one (keeps original created_at)."""
    with _app_lock:
        app_conn.execute(
            """
            INSERT INTO chat_titles (thread_id, title, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title
            """,
            (str(thread_id), title, time.time()),
        )
        app_conn.commit()


def rename_thread_title(thread_id, title: str):
    with _app_lock:
        app_conn.execute(
            "UPDATE chat_titles SET title = ? WHERE thread_id = ?",
            (title, str(thread_id)),
        )
        app_conn.commit()


def retrieve_all_titles() -> dict:
    with _app_lock:
        rows = app_conn.execute("SELECT thread_id, title FROM chat_titles").fetchall()
    return {row[0]: row[1] for row in rows}


def retrieve_all_threads() -> list[str]:
    """Thread IDs ordered oldest -> newest. Reads only the small titles table (no checkpoint scan)."""
    with _app_lock:
        rows = app_conn.execute(
            "SELECT thread_id FROM chat_titles ORDER BY created_at ASC, rowid ASC"
        ).fetchall()
    return [row[0] for row in rows]


def delete_thread_data(thread_id):
    """Permanently delete a conversation: its checkpoints and its title."""
    thread_id = str(thread_id)
    with _app_lock:
        if hasattr(checkpointer, "delete_thread"):
            checkpointer.delete_thread(thread_id)
        else:
            # Fallback for older checkpointer versions
            for table in ("checkpoints", "writes"):
                app_conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        app_conn.execute("DELETE FROM chat_titles WHERE thread_id = ?", (thread_id,))
        app_conn.commit()


setup_tables()