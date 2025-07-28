import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embeddings import get_embeddings
from langchain_community.vectorstores import Chroma


chromapath = "chroma"
datapath = r"D:\AI_Projects\RAG(PDF)\data"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action = "store_true", help="Reset the Database")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database

    documents = load_documents()
    chunks = split_document(documents)
    add_to_chroma(chunks)
    




def load_documents():
    document_loader = PyPDFDirectoryLoader(datapath)
    return document_loader.load()

def split_document(documents : list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800, 
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False,
        )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks : list[Document]):
    db = Chroma(persist_directory = chromapath, embedding_function=get_embeddings())
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"No.of existng documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]

        db.add_documents(new_chunks, ids=new_chunks_ids)
        #db.persist()
    else:
        print("✅ No new documents to add")
        
def calculate_chunk_ids(chunks):

    last_page_id = None,
    current_chunk_index = 0,


    for chunk in chunks:
        source = chunk.metadata.get("source"),
        page = chunk.metadata.get("page"),
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id} : {current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(chromapath):
        shutil.rmtree(chromapath)
                      
                      
if __name__ == "__main__":
    main()

        


    




