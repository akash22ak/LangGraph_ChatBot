import sqlite3
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

class ChatState(TypedDict):
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
graph.add_edge("chat_node",  END)

chatbot = graph.compile(checkpointer=checkpointer)


# ── Title persistence ──────────────────────────────────────────

def setup_titles_table():
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL)
    """)
    conn.commit()

def save_thread_title(thread_id: str, title: str):
    conn.execute("INSERT INTO chat_titles (thread_id, title) VALUES (?, ?)", (str(thread_id), title))
    conn.commit()

def retrieve_all_titles() -> dict:
    cursor = conn.execute("SELECT thread_id, title FROM chat_titles")
    return {row[0]: row[1] for row in cursor.fetchall()}

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

setup_titles_table()
