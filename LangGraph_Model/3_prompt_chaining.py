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

class BlogState(TypedDict):

    topic: str
    outline: str
    content: str


def generate_outline(state: BlogState) -> BlogState:

    topic = state['topic']

    prompt = f'generate outline for the blog on the topic - {topic}'

    outline = model.invoke(prompt).content

    state['outline'] = outline

    return state



def generate_blog(state: BlogState) -> BlogState:

    topic = state['topic']
    outline = state['outline']

    prompt = f'create a blog on the topic - {topic} with the help of given following outline \n {outline}'

    content = model.invoke(prompt).content

    state['content'] = content

    return state


graph = StateGraph(BlogState);

graph.add_node('generate_outline', generate_outline)
graph.add_node('generate_blog', generate_blog)

graph.add_edge(START, 'generate_outline')
graph.add_edge('generate_outline', 'generate_blog')
graph.add_edge('generate_blog', END)

workflow = graph.compile()

initial_state = {'topic': 'Brain & Beauty'}
final_state = workflow.invoke(initial_state)

print(final_state['outline'])
print(final_state['content'])


png = workflow.get_graph().draw_mermaid_png()
with open("workflow3.png", "wb") as f:
    f.write(png)

print("Saved workflow3.png")
