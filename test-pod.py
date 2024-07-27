from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain_community.chat_models import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class CustomChatOllama(ChatOllama):
    def __init__(self, temperature, model, base_url):
        super().__init__(temperature=temperature, model=model, base_url=base_url)

    def generate_response(self, prompt):
        headers = {
            "Authorization": f"Bearer {os.getenv('POD_API_KEY')}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json={"input": {"method_name": "generate", "input": {"prompt": prompt}}}
        )
        if response.status_code == 200:
            return response.json()["output"]["response"]
        else:
            raise Exception(f"Failed to fetch response: {response.status_code} - {response.text}")

llm = CustomChatOllama(
    temperature=0,
    model="mistral",
    base_url="https://api.runpod.ai/v2/yuyxza228rw6tx/runsync",
)

app = FastAPI()

def chat_with_document(document_text, prompt):
    try:
        # Create a prompt template for the chat model
        system_message = SystemMessagePromptTemplate.from_template(
            "You are an AI chatbot by that goes by the name Agba, you typically act as a chatbot for querying the newspaper knowledge that a user is viewing and you only ever provide information to users who ask you a question based on the given document. When asked questions about your identity, Say you are Agba, an AI chatbot made by the innovation team at CJID. Be sure to always chat with users in a friendly researcher-like tone and always a lot of detail based on the question asked by a user on the document. Note that the document is a knowledge bank for you to answer user question. Here is the document - {document}. When met with any salutation message, respond with a salutation and your purpose. At the end of each response, and on a new line let users know that they can provide feedback about the response by sending an email to monsur@thecjid.org or tobi.ekudayo@thecjid.org. Be sure to have the mails as hyperlinks. Also, do not hallucinate"
        )
        human_message = HumanMessagePromptTemplate.from_template("{prompt}")

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        # Generate the response using the chat model
        formatted_prompt = chat_prompt.format_prompt(document=document_text, prompt=prompt).to_string()
        response = llm.generate_response(formatted_prompt)
        return response
    except Exception as e:
        return f"Error: {e}"

@app.post("/chat-with-document/")
async def chat_with_document_endpoint(extracted_text: str = Form(...), query: str = Form(...)):
    try:
        result = chat_with_document(extracted_text, query)
        # print(extracted_text)
        return JSONResponse(content={"response": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
