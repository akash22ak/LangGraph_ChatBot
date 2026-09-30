import os
import time
import sqlite3
import requests
import threading
from datetime import date
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from ddgs import DDGS
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

DB_PATH = "ChatBot.DB"

# ── Database connections ──────────────────────────────────────────────────────────────────
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


# ── LLM ──────────────────────────────────────────────────────────────────────────────

llm = HuggingFaceEndpoint(
    # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    repo_id="Qwen/Qwen3.8-2.4T-A95B",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


# ── Tools ──────────────────────────────────────────────────────────────────────────────────

@tool
def search_tool(query: str) -> str:
    """Search the web (DuckDuckGo) for current information. Returns titles, snippets and URLs."""
    try:
        results = DDGS().text(query, region="wt-wt", max_results=5)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No search results found."

    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}"
        for i, r in enumerate(results, 1)
    )


@tool
def calculator(first_no: float, second_no: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_no + second_no
        elif operation == "sub":
            result = first_no - second_no
        elif operation == "mul":
            result = first_no * second_no
        elif operation == "div":
            if second_no == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_no / second_no
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_no, "second_num": second_no, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}



@tool
def get_stock_price(symbol: str) -> dict:
    """Get the latest stock price for a ticker symbol such as 'AAPL' or 'TSLA'."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY is not set"}
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol.upper(), "apikey": api_key},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}

    quote = data.get("Global Quote")
    if not quote:
        # Alpha Vantage returns HTTP 200 with a "Note"/"Information" field when rate-limited
        return {"error": data.get("Note") or data.get("Information") or f"No data for {symbol}"}

    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "change_percent": quote.get("10. change percent"),
        "latest_trading_day": quote.get("07. latest trading day"),
    }


# Tool List
tools = [search_tool, calculator, get_stock_price]

# LLM tool-aware
model_with_tools = model.bind_tools(tools)


# ── LangGraph ──────────────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that automatically decides whether to use web search and calculation tools."""
    messages = state["messages"]
    
    system_instruction = SystemMessage(content=(
        f"You are a helpful assistant. Today's date is {date.today():%B %d, %Y}. "
        "Answer directly from your own knowledge for greetings, general knowledge and reasoning. "
        "Use `calculator` for arithmetic, `get_stock_price` for stock prices, and `search_tool` "
        "for recent events or facts you are unsure about. "
        "Never invent facts or tool results. If a tool returns an error, tell the user."
    ))
    
    # Prepend the system prompt to the conversation thread history
    full_messages = [system_instruction] + messages

    try:
        response = model_with_tools.invoke(full_messages)
    except Exception:
        response = model.invoke(full_messages)

    if response.tool_calls:
        response.content = ""      # Tool execution turns remain quiet on the frontend UI
    return {"messages": [response]}



tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

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