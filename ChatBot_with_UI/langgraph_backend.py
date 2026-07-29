from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
load_dotenv("/home/srestha/Desktop/Langraph/.env")


class ChatState(TypedDict):

    message: Annotated[list[BaseMessage], add_messages]


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

def chat_node(state: ChatState):

    # take user query from state
    message = state["message"]
    # send it to model
    response = model.invoke(message)
    # store result in state
    return {"message": [response]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node",  END)

chatbot = graph.compile(checkpointer=checkpointer)
