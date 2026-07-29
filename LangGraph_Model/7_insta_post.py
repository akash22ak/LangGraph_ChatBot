from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
import json


load_dotenv()

llm1 = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

generator_model = ChatHuggingFace(llm=llm2)
evaluator_model = ChatHuggingFace(llm=llm1)
optimiser_model = ChatHuggingFace(llm=llm1)

class PostState(TypedDict):

    topic: str
    post: str
    evaluate: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int


class PostEvaluation (BaseModel):

    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="Constructive feedback for the post.")


def generate_post(state: PostState):

    # prompt
    messages = [
        SystemMessage(content="You are a funny and clever Instagram influencer."),
        HumanMessage(content=f"""
    Write a short, original, and hilarious post on the topic: "{state['topic']}".
    Rules:
    - Do NOT use question-answer format.
    - Max 280 characters.
    - Use observational humor, irony, sarcasm, or cultural references.
    - Think in meme logic, punchlines, or relatable takes.
    - Use simple, day to day english
    - This is version {state['iteration'] + 1}.
    """)
    ]

    response = generator_model.invoke(messages).content

    return {"post": response}


def evaluation_results(post: str) -> dict:

    post = post
    #prompt
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Instagram critic. You evaluate instagram post based on humor, originality, virality, and post format."),
        HumanMessage(content=f"""
                     Evaluate the following post:
        Post: "{post}"

        Use the criteria below to evaluate the post:
        1. Originality - Is this fresh, or have you seen it a hundred times before?
        2. Humor Did it genuinely make you smile, laugh, or chuckle?
        3. Punchiness Is it short, sharp, and scroll-stopping?
        4. Virality Potential - Would people repost or share it?
        5. Format - Is it a well-formed post (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

        Auto-reject if:
        - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
        - It exceeds 280 characters
        - It reads like a traditional setup-punchline joke
        - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

        ### Respond ONLY in structured format:
        - evaluation: "approved" or "needs_improvement"
        - feedback: One paragraph explaining the strengths and weaknesses
        """)
    ]

    response = evaluator_model.invoke(messages).content

    data = json.loads(response)

    result = PostEvaluation.model_validate(data)

    return result.model_dump()


def evaluate_post(state: PostState):

    response = evaluation_results(state["post"])

    return {"evaluate": response["evaluate"], "feedback": response["feedback"]}


def optimise_post(state: PostState):

    messages = [
    SystemMessage(content="You punch up post for virality and humor based on given feedback."),
    HumanMessage(content=f"""
    Improve the post based on this feedback:
    "{state['feedback']}"

    Topic: "{state['topic']}"
    Original Post:
    {state['post']}

    Re-write it as a short, viral-worthy post. Avoid Q&A style and stay under 280 characters.
    """)
    ]

    response = optimiser_model.invoke(messages).content
    iteration = state["iteration"] + 1

    return {"post": response, "iteration": iteration}


def route_evaluation(state: PostState):

    if state["evaluate"] == "approved" or state["iteration"] >= state["max_iteration"]:
        return "approved"
    else:
        return "needs_improvement"


graph = StateGraph(PostState)

graph.add_node("generate", generate_post)
graph.add_node("evaluate", evaluate_post)
graph.add_node("optimise", optimise_post)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", route_evaluation, {"approved": END, "needs_improvement": "optimise"})
graph.add_edge("optimise", "evaluate")

workflow = graph.compile()

initial_state = {
    "topic": "Corruption",
    "iteration": 1,
    "max_iteration": 5
    }
final_state = workflow.invoke(initial_state)

print(final_state)


png = workflow.get_graph().draw_mermaid_png()
with open("workflow7.png", "wb") as f:
    f.write(png)

print("Saved workflow7.png")