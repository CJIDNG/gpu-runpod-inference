from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain_community.chat_models import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the API key for OpenAI from environment variables
# openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI with the API key
llm = ChatOllama(temperature=0, model="mistral")

app = FastAPI()

def chat_with_document(document_text, prompt):
    try:
        # Create a prompt template for the chat model
        system_message = SystemMessagePromptTemplate.from_template("You are an AI chatbot by that goes by the name Agba, you typically act as a chatbot for querying the newspaper that a user is viewing and you only ever provide information to users who ask you a question based on the given document. When asked questions about your identity, Say you are Agba, an AI chatbot made by the innovation team at CJID. Be sure to always chat with users in a friendly researher like tone and always alot of detail based on the question asked by a user on the document. Note that the document is a knowledge bank for you to answer user question. Here is the document - {document}")
        human_message = HumanMessagePromptTemplate.from_template("{prompt}")

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        # Generate the response using the chat model
        response = llm(chat_prompt.format_prompt(document=document_text, prompt=prompt).to_messages())
        return response.content
    except Exception as e:
        return f"Error: {e}"

@app.post("/chat-with-document/")
async def chat_with_document_endpoint(extracted_text: str = Form(...), query: str = Form(...)):
    try:
        result = chat_with_document(extracted_text, query)
        return JSONResponse(content={"response": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
