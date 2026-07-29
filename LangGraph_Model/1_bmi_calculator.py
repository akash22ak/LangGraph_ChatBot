from langgraph.graph import StateGraph, START, END
from typing import TypedDict, cast


# define state
class BMIState(TypedDict):

    weight_kg: float
    height_m: float
    bmi: float
    category: str

# create graph
graph = StateGraph(BMIState)

# define the node function
def calculate_bmi(state: BMIState) -> BMIState:

    weight = state['weight_kg']
    height = state['height_m']

    bmi = weight/(height**2)

    state['bmi'] = round(bmi, 2)

    return state


def lable_bmi(state: BMIState) -> BMIState:

    bmi = state['bmi']

    if bmi < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["category"] = "Normal"
    elif 25 <= bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"

    return state

# create nodes
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('lable_bmi', lable_bmi)

# create edges
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'lable_bmi')
graph.add_edge('lable_bmi', END)

# compile the graph
workflow = graph.compile()

# execute the graph
initial_state = cast(BMIState, {
    'weight_kg': 79, 
    'height_m': 1.6
})
final_state = workflow.invoke(initial_state)

print(final_state)


png = workflow.get_graph().draw_mermaid_png()
with open("workflow.png", "wb") as f:
    f.write(png)

print("Saved workflow.png")