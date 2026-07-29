from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
import json


load_dotenv()

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class ReviewState(TypedDict):

    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


class sentimentSchema(BaseModel):

    sentiment: Literal["positive", "negative"] = Field(description= 'give the semtiment of review')


class Diagnosis_Output (BaseModel):

    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field (description='How urgent or critical the issue appears to be')

def get_sentiment(review: str) -> str:

    prompt = f"""
    Return ONLY valid JSON.
    {{
        "sentiment": "positive" or "negative"
    }}
    Review:
    {review}
    """
    response = model.invoke(prompt)
    
    #convert the model output into json format
    data = json.loads(response.content)

    # validate the json using pydentic
    result = sentimentSchema.model_validate(data)

    return result.sentiment


def find_sentiment(state: ReviewState):
    sentiment = get_sentiment(state["review"])

    return {"sentiment": sentiment}


def check_sentiment(state: ReviewState) -> Literal["run_diagnosis", "positive_response"]:

    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"
    

def positive_response(state: ReviewState):

    prompt = f'write a positive and thankfull reply for the review: \n - {state["review"]}'

    ps = model.invoke(prompt).content

    return {"response": ps}


def get_diagnosis_report(review: str) -> dict:

    prompt = f"""
    Return ONLY valid JSON.

    {{
        "issue_type": "UX" | "Performance" | "Bug" | "Support" | "Other",
        "tone": "angry" | "frustrated" | "disappointed" | "calm",
        "urgency": "low" | "medium" | "high"
    }}

    Review:
    {review}
    """

    response = model.invoke(prompt)

    data = json.loads(response.content)

    result = Diagnosis_Output.model_validate(data)

    return result.model_dump()


def run_diagnosis(state: ReviewState):

    diagnosis = get_diagnosis_report(state["review"])

    return {"diagnosis": diagnosis}


def negative_response(state: ReviewState):

    diagnosis = state['diagnosis']

    prompt = f'''You are a support assistant.
    The user had a {diagnosis["issue_type"]} issue, sounded {diagnosis["tone"]}, and the issue urgency level is {diagnosis["urgency"]}.
    Write an emphathetic and helpull reply.
    '''

    ns = model.invoke(prompt).content

    return {"response": ns}
    

graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)


graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)
graph.add_edge("positive_response", END)


workflow = graph.compile()

initial_state = {"review": """I've been using this app for about a month now, and I must say, the user interface is incredibly clean and intuitive.
Everything is exactly where you'd expect it to be. It's rare to find something that just works without needing a tutorial.
Great job to the design team!"""}
final_state = workflow.invoke(initial_state)

print(final_state)



png = workflow.get_graph().draw_mermaid_png()
with open("workflow6.png", "wb") as f:
    f.write(png)

print("Saved workflow6.png")