# Imports

import os
import json
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_community.document_loaders import  TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# API Key
os.environ['OPENAI_API_KEY'] = <your-api-key>

# Paths
CHROMA_PATH='../chroma'
DATA_PATH='../data/'

#### INDEXING ####

# Load Documents
txt_paths = [
    DATA_PATH + 'sample1.txt',
    DATA_PATH + 'sample2.txt',
    DATA_PATH + 'sample3.txt'
]

json_paths = [
    DATA_PATH + 'sample1.json',
    DATA_PATH + 'sample2.json',
    DATA_PATH + 'sample3.json'
]

txts = [TextLoader(path).load() for path in txt_paths]
txt_docs = [item for sublist in txts for item in sublist]

json_docs = []
for path in json_paths:
    with open(path, 'r', encoding='utf-8') as file:
        json_docs += [json.load(file)]

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=0
)
txt_chunks = text_splitter.split_documents(txt_docs)

json_splitter = RecursiveJsonSplitter(
    max_chunk_size=2000
)
json_chunks = json_splitter.create_documents(json_docs, ensure_ascii=False)

chunks = txt_chunks + json_chunks

# Embed
vectorstore = Chroma.from_documents(documents=chunks, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")