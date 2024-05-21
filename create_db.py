from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
import json
import os
import shutil

CHROMA_PATH='chroma'
DATA_PATH='data'

def main():
    make_data_store()

def make_data_store():
    txt_documents, json_documents = load_documents()
    txt_chunks = split_txt(txt_documents)
    #json_chunks = split_json(json_documents)
    #save_to_chroma(txt_chunks, json_chunks)
    save_to_chroma(txt_chunks, json_documents)

def load_documents():
    txt_loader = DirectoryLoader(
    path=DATA_PATH,
    glob="*.txt",
    loader_cls=TextLoader,
    show_progress=True
    )
    txt_documents = txt_loader.load()
    print(f"Load {len(txt_documents)} txt files.")
    
    json_loader = DirectoryLoader(
        path=DATA_PATH,
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={'jq_schema': '.', 'text_content': False},
        show_progress=True
    )
    json_documents = json_loader.load()
    print(f"Load {len(json_documents)} json files.")
    
    return txt_documents, json_documents

def split_txt(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} txt files into {len(chunks)} chunks.")
    
    document = chunks[1]
    print(document.page_content)
    print(document.metadata)
    
    return chunks

def split_json(json_files):
    splitter = RecursiveJsonSplitter(max_chunk_size=300)
    print(len(json_files))
    documents = splitter.create_documents(json_files, convert_lists=False, metadatas=None)
    print(len(documents))
    chunks = splitter.split_json(documents)
    print(f"Split {len(json_files)} json files into {len(chunks)} chunks.")
    
    return chunks

def save_to_chroma(txt_chunks, json_chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    txt_db = Chroma.from_documents(
        txt_chunks, OpenAIEmbeddings(openai_api_key=''), persist_directory=CHROMA_PATH
    )
    txt_db.persist()
    print(f"Saved {len(txt_chunks)} text chunks to {CHROMA_PATH}.")
    
    json_db = Chroma.from_documents(
        [json.dumps(chunk) for chunk in json_chunks], 
        OpenAIEmbeddings(openai_api_key=''), 
        persist_directory=CHROMA_PATH
    )
    json_db.persist()
    print(f"Saved {len(json_chunks)} json chunks to {CHROMA_PATH}.")
    
if __name__ == "__main__":
    main()