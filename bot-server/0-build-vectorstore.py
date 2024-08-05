#!/usr/bin/python3

import time, os, sys, queue, re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

MODELS_BASE_PATH='/home/ubuntu/models/'
os.environ['HF_HOME'] = MODELS_BASE_PATH
KBREPO_PATH="../cb-kb-docs"
VECTORSTORE_PATH="./"
VECTORSTORE_FILE="LangChain_FAISS"

################################################
print("Load KB text from repo files...")


# Define a dictionary to map file extensions to their respective loaders
loaders = {
    '.json': JSONLoader,
    '.txt': TextLoader,
    '.csv': CSVLoader,
	'.pdf': PyPDFLoader,
}

# Define a function to create a DirectoryLoader for a specific file type
def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
    )

# Get documents using appropriate loader
docs = []
docs += create_directory_loader('.pdf', KBREPO_PATH).load()
docs += create_directory_loader('.txt', KBREPO_PATH).load()
docs += create_directory_loader('.csv', KBREPO_PATH).load()
docs += create_directory_loader('.pdf', KBREPO_PATH).load()


################################################
print("Split text into chunks...")
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(docs)

################################################
print("Create vectorstore of doc text embeddings (~46 secs)...")
start=time.time()

# from HF
#model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = MODELS_BASE_PATH + "all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}   # cuda: ~46 secs, cpu:~10 mins
encode_kwargs = {'normalize_embeddings': False}
hf_embedder = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = FAISS.from_documents(documents=all_splits, embedding=hf_embedder)
print("Time Taken --> ", time.time()-start) 

vectorstore.save_local(VECTORSTORE_PATH, VECTORSTORE_FILE)
