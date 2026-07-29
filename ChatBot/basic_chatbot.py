from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

class ChatState(TypedDict):

    message: Annotated[list[BaseMessage], add_messages]


llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
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


checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node",  END)

chatbot = graph.compile(checkpointer=checkpointer)

initial_state = {
    "message": [HumanMessage(content="What is the capital of Rajasthan?")]
}

# final_state = chatbot.invoke(initial_state)

# print(final_state)
# print(final_state['message'][-1].content)

thread_id = '1'
while True:

    user_input = input("Type here: ")
    print('User: ', user_input)

    if user_input.strip().lower() in ['exit', 'quit', 'end', 'bye']:
        break

    config = {'configurable': {'thread_id': thread_id}}
    response = chatbot.invoke({'message': [HumanMessage(content=user_input)]}, config=config)
    print('AI: ', response['message'][-1].content)

state = chatbot.get_state(config=config)
print(state)