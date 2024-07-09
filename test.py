import streamlit as st # type: ignore
from langchain_community.vectorstores import Weaviate # type: ignore
import weaviate # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain.prompts import ChatPromptTemplate # type: ignore
from langchain import HuggingFaceHub # type: ignore
from langchain.schema.runnable import RunnablePassthrough # type: ignore
from langchain.schema.output_parser import StrOutputParser  # type: ignore
from weaviate.client import WeaviateClient # type: ignore
from weaviate.auth import AuthApiKey # type: ignore
from langchain.document_loaders import PyPDFLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain import HuggingFaceHub # type: ignore
from langchain.schema.runnable import RunnablePassthrough # type: ignore
from langchain.schema.output_parser import StrOutputParser # type: ignore
import locale
import os



# Set up your Weaviate client
WEAVIATE_CLUSTER = "https://my-test-l7tfbws3.weaviate.network"
WEAVIATE_API_KEY = "quM2cpPBOXkgKSFOio1q00a315eloG4mb83g"

WEAVIATE_URL = WEAVIATE_CLUSTER
WEAVIATE_API_KEY = WEAVIATE_API_KEY

client = weaviate.Client(
    url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)

# fixing unicode error in google colab
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# specify embedding model (using huggingface sentence transformer)
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
#model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(
  model_name=embedding_model_name,
  #model_kwargs=model_kwargs
)

# Loading pdf files 
loader = PyPDFLoader("C:/SANJAY R/Text File/Nissan Magnite Manual.pdf", extract_images=False)
pages = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(pages)

# Creating Vector Database
vector_db = Weaviate.from_documents(
    docs, embeddings, client=client, by_text=False
)

# AI agent Instructions template
template="""You are an AI assistant integrated into a vehicle's system. Your primary function is to help the user operate and maintain the car safely and efficiently. You have access to the following resources:

The car's user manual
Real-time data from various sensors and systems in the vehicle

Your responsibilities include:

Providing clear, step-by-step instructions for any car-related task the user requests.
Answering questions about the vehicle's features, functions, and maintenance.
Monitoring the user's actions through sensor data and providing immediate feedback.
Alerting the user to any potential issues or safety concerns.

When interacting with the user:

Always prioritize safety. If a user's request or action could compromise safety, inform them immediately and suggest safer alternatives.
Provide concise initial responses, but offer to elaborate if the user needs more detail.
Use clear, non-technical language whenever possible. If technical terms are necessary, explain them.
When giving instructions, break them down into simple steps. Confirm the user's completion of each step before moving to the next.
If sensor data indicates the user has performed an action incorrectly, provide immediate, constructive feedback. For example:
"I've noticed the engine temperature is rising unusually quickly. Let's double-check that the coolant was filled to the correct level."
Anticipate potential issues based on the car's condition and the user's actions. Offer proactive advice when appropriate.
If the user seems confused or frustrated, offer to explain things in a different way or provide additional context from the user manual.
For complex procedures, offer to guide the user through the process step-by-step, waiting for confirmation before proceeding to the next step.
If a task is beyond the user's capabilities or requires professional assistance, advise them to seek help from a qualified mechanic.
Stay updated on the vehicle's status and inform the user of any necessary maintenance or potential issues.
Question: {question}
Context: {context}
Answer:
"""

prompt=ChatPromptTemplate.from_template(template)

# Set up your HuggingFace model
huggingfacehub_api_token = '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'
model = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 1, "max_length": 180}
)


output_parser=StrOutputParser()

# Retrive Vector Database
retriever=vector_db.as_retriever()


rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)


# Streamlit UI
st.header("Vehicle Assistant Chatbot 💬 📚")
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "What would you like to know about your vehicle?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me something.."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({prompt})

    response = rag_chain.invoke(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({response})
