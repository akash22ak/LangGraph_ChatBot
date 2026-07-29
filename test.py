from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the capital of India")  
print(response.content)





# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv ( )

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# result = model.invoke("What is the capital of India?")

# print(result.content)




