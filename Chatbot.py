#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import getpass
import os
import requests
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API Key: ")

if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter LangChain API Key: ")

llm = ChatOpenAI(model="gpt-4", temperature=0)

pdf_url = "https://chatbotpdfdocumentation.s3.us-east-1.amazonaws.com/AI+Engineer.pdf"

with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
    response = requests.get(pdf_url)
    temp_pdf.write(response.content)
    temp_pdf_path = temp_pdf.name

loader = PyMuPDFLoader(temp_pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_prompt = (
    "You are an AI assistant that provides concise and accurate answers about Promtior. "
    "Use the retrieved context to answer the question. If you don't know the answer, say that you don't know."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("Ask any question about Promtior. Type 'exit' to end the chat.")

while True:
    user_question = input("You: ")
    if user_question.lower() == 'exit':
        print("Ending the chat. Goodbye!")
        break

    response = rag_chain.invoke({"input": user_question})
    print(f"AI: {response['answer']}\n")

