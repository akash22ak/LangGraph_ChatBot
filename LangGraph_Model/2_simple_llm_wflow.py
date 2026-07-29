from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict


load_dotenv()

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class LLMState(TypedDict):

    question: str
    answer: str

def llm_question(state: LLMState) -> LLMState:

    question = state['question']
    prompt = f'Answer the following question {question}'
    answer = model.invoke(prompt).content
    state["answer"] = answer

    return state


graph = StateGraph(LLMState)

graph.add_node('llm_question', llm_question)

graph.add_edge(START, 'llm_question')
graph.add_edge('llm_question', END)

workflow = graph.compile()

initial_state = {'question': 'what is name of missile man of india'}
final_state = workflow.invoke(initial_state)

print(final_state['answer'])
    
png = workflow.get_graph().draw_mermaid_png()
with open("workflow2.png", "wb") as f:
    f.write(png)

print("Saved workflow2.png")