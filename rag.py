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

# Configurable variables
openai_api_key = os.environ.get('OPENAI_API_KEY')
model_name = os.environ.get('MODEL_NAME')
temperature = os.environ.get('TEMPERATURE')
chunk_size = os.environ.get('CHUNK_SIZE')
data_path = os.environ.get('DATA_PATH')

#### INDEXING ####

# Load Documents
txt_paths = [
    data_path + 'sample1.txt',
    data_path + 'sample2.txt',
    data_path + 'sample3.txt'
]

json_paths = [
    data_path + 'sample1.json',
    data_path + 'sample2.json',
    data_path + 'sample3.json'
]

txts = [TextLoader(path).load() for path in txt_paths]
txt_docs = [item for sublist in txts for item in sublist]

json_docs = []
for path in json_paths:
    with open(path, 'r', encoding='utf-8') as file:
        json_docs += [json.load(file)]

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size
)
txt_chunks = text_splitter.split_documents(txt_docs)

json_splitter = RecursiveJsonSplitter(
    max_chunk_size=chunk_size
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
llm = ChatOpenAI(model_name=model_name, temperature=temperature)

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