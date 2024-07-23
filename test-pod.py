from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain_community.chat_models import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class CustomChatOllama(ChatOllama):
    def __init__(self, temperature, model, base_url, api_key):
        super().__init__(temperature=temperature, model=model, base_url=base_url)
        self.api_key = api_key

    def generate_response(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json={"input": {"method_name": "generate", "input": {"prompt": messages}}}
        )
        if response.status_code == 200:
            return response.json()["output"]["response"]
        else:
            raise Exception(f"Failed to fetch response: {response.status_code} - {response.text}")

llm = CustomChatOllama(
    temperature=0,
    model="mistral",
    base_url="https://hhjkd8tu4nyakv-11434.proxy.runpod.net/",
    api_key=os.getenv['POD_API_KEY']
)

app = FastAPI()

def chat_with_document(document_text, prompt):
    try:
        # Create a prompt template for the chat model
        system_message = SystemMessagePromptTemplate.from_template(
            "You are an AI chatbot by that goes by the name Agba, you typically act as a chatbot for querying the newspaper knowledge that a user is viewing and you only ever provide information to users who ask you a question based on the given document. When asked questions about your identity, Say you are Agba, an AI chatbot made by the innovation team at CJID. Be sure to always chat with users in a friendly researcher-like tone and always a lot of detail based on the question asked by a user on the document. Note that the document is a knowledge bank for you to answer user question. Here is the document - {document}. When met with any salutation message, respond with a salutation and your purpose"
        )
        human_message = HumanMessagePromptTemplate.from_template("{prompt}")

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        # Generate the response using the chat model
        response = llm.generate_response(chat_prompt.format_prompt(document=document_text, prompt=prompt).to_messages())
        return response
    except Exception as e:
        return f"Error: {e}"

@app.post("/chat-with-document/")
async def chat_with_document_endpoint(extracted_text: str = Form(...), query: str = Form(...)):
    try:
        result = chat_with_document(extracted_text, query)
        return JSONResponse(content={"response": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
