import time
import sqlite3
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


class ChatState(TypedDict):
    # NOTE: key kept as "message" so existing threads in ChatBot.DB keep working.
    message: Annotated[list[BaseMessage], add_messages]


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


def chat_node(state: ChatState):
    message = state["message"]
    response = model.invoke(message)
    return {"message": [response]}


conn = sqlite3.connect(database="ChatBot.DB", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ── Title persistence ──────────────────────────────────────────

def setup_titles_table():
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL DEFAULT 0
        )
    """)

    # Migration: older DBs were created without created_at
    columns = [row[1] for row in conn.execute("PRAGMA table_info(chat_titles)")]
    if "created_at" not in columns:
        conn.execute("ALTER TABLE chat_titles ADD COLUMN created_at REAL NOT NULL DEFAULT 0")

    conn.commit()


def save_thread_title(thread_id, title: str):
    """Insert a title, or update it if the thread already has one (keeps original created_at)."""
    conn.execute(
        """
        INSERT INTO chat_titles (thread_id, title, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title
        """,
        (str(thread_id), title, time.time()),
    )
    conn.commit()


def retrieve_all_titles() -> dict:
    cursor = conn.execute("SELECT thread_id, title FROM chat_titles")
    return {row[0]: row[1] for row in cursor.fetchall()}


def retrieve_all_threads() -> list[str]:
    """
    Returns thread IDs (as strings) ordered oldest -> newest.
    - Threads with a title are ordered by created_at.
    - Orphaned threads (checkpoint exists but no title row) are placed first (treated as oldest).
    """
    titled = [
        row[0]
        for row in conn.execute(
            "SELECT thread_id FROM chat_titles ORDER BY created_at ASC, rowid ASC"
        )
    ]
    titled_set = set(titled)

    orphans = set()
    for checkpoint in checkpointer.list(None):
        tid = str(checkpoint.config['configurable']['thread_id'])
        if tid not in titled_set:
            orphans.add(tid)

    return sorted(orphans) + titled


setup_titles_table()