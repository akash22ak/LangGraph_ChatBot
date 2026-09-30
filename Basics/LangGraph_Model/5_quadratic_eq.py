from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class QuadState(TypedDict):

    a: int
    b: int
    c: int

    equation: str
    discriminant: float
    result: str


def form_equation(state: QuadState):

    a = state['a']
    b = state['b']
    c = state['c']
    eq = (
        f"{a}x² "
        f"{'+' if b >= 0 else '-'} {abs(b)}x "
        f"{'+' if c >= 0 else '-'} {abs(c)}"
    )

    return {'equation': eq}


def calculate_discriminant(state: QuadState):

    a = state['a']
    b = state['b']
    c = state['c']
    d = b**2 - 4*a*c

    return {'discriminant': d}


def real_roots(state: QuadState):

    a = state['a']
    b = state['b']
    d = state['discriminant']
    r1 = (-b + d)/2*a
    r2 = (-b - d)/2*a

    result = f'The roots are r1: {r1} and r2: {r2}'

    return {'result': result}


def repeated_roots(state: QuadState):

    a = state['a']
    b = state['b']
    r = (-b)/2*a

    result = f'The roots are r1: {r} and r2: {r}'

    return {'result': result}


def no_real_roots(state: QuadState):

    result = f'Their is no real roots'

    return {'result': result}


def check_d(state:QuadState) -> Literal['real_roots', 'repeated_roots', 'no_real_roots']:

    d = state['discriminant']
    
    if d > 0:
        return 'real_roots'
    elif d == 0:
        return 'repeated_roots'
    else:
        return 'no_real_roots'

graph = StateGraph(QuadState)

graph.add_node('form_equation', form_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

graph.add_edge(START, 'form_equation')
graph.add_edge('form_equation', 'calculate_discriminant')

graph.add_conditional_edges('calculate_discriminant', check_d)

graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

workflow = graph.compile()

initial_state = {
    'a': 1,
    'b': 2,
    'c': 1
}

final_state = workflow.invoke(initial_state)
print(final_state)



png = workflow.get_graph().draw_mermaid_png()
with open("workflow5.png", "wb") as f:
    f.write(png)

print("Saved workflow5.png")