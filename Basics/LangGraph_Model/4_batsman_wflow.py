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


class BatsmanState(TypedDict):

    runs: int
    balls: int
    fours: int
    sixes: int

    sr: float
    bpb: float
    boundary_percentage: float
    summary: str


def calculate_sr(state: BatsmanState):

    sr = (state['runs']/state['balls'])*100
    state['sr'] = sr
    return {'sr': sr}

def calculate_bpb(state: BatsmanState):

    bpb = state['balls']/(state['fours'] + state['sixes'])
    state['bpb'] = bpb
    return {'bpb': bpb}

def calculate_boundary_percentage(state: BatsmanState):

    bp = (((state['sixes'] * 6) + (state['fours'] * 4))/(state['runs']))*100 
    state['boundary_percentage'] = bp
    return {'boundary_percentage': bp}

def summary(state: BatsmanState):

    summary = f"""
Strike Rate - {state['sr']} \n
Balls per boundary - {state['bpb']} \n
Boundary percentage - {state['boundary_percentage']}
"""
    state['summary'] = summary
    return state


graph = StateGraph(BatsmanState)

graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percentage', calculate_boundary_percentage)
graph.add_node('summary', summary)

graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percentage')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percentage', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()

initials_state = {
    'runs': 100,
    'balls': 50, 
    'fours': 6,
    'sixes': 4
}

final_state = workflow.invoke(initials_state)
print(final_state['summary'])

png = workflow.get_graph().draw_mermaid_png()
with open("workflow4.png", "wb") as f:
    f.write(png)

print("Saved workflow4.png")