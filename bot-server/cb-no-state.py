#!/usr/bin/python3

# Import necessary libraries
import os
import numpy as np
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LOGGING ============================================
import logging
logfile = "./logs/consultabot.log"
loglevel = logging.DEBUG
logfmode = 'w'                # w = overwrite, a = append
logging.basicConfig(filename=logfile, encoding='utf-8', level=loglevel, filemode=logfmode)

# CONSTANTS ============================================
#LLM_MODEL_PATH = "./llama-2-13b-chat.Q5_K_M.gguf")
#LLM_MODEL_PATH = "./Llama-3-Instruct-8B-SPPO-Iter3-Q5_K_M.gguf"
MODELS_BASE_PATH='/home/ubuntu/models/'
os.environ['HF_HOME'] = MODELS_BASE_PATH
LLM_MODEL_PATH = MODELS_BASE_PATH + "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
VECTORSTORE_PATH="./"
VECTORSTORE_FILE="LangChain_FAISS"

################################################
# Load vector store from disk
def loadVectorStore():
    print("\nRead vectorstore from disk...")

    model_name = MODELS_BASE_PATH + "all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    _embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    _vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        _embedder,
        VECTORSTORE_FILE,
        allow_dangerous_deserialization=True,  # we trust it because we created it
    )
    return _vectorstore

################################################
# returns True if query returns k entries all <= with max_score
def validate_query_relevance(_vectorstore, query_text, k=5, max_score=1):
    # returns list of (Document,score) tuples, lower score == more similar
    docs = _vectorstore.similarity_search_with_score(
        query=query_text,
        k=k,
    )
    for i in range(0,len(docs)):
        print(docs[i][1])
    # return True if all scores <= max_score
    return all(score <= max_score for (_, score) in docs)

################################################
# Load LLM from disk
def loadLLM():
    print("\nLoading LLM from disk...")

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    temp=0              # stick to the facts
    n_gpu_layers = -1   # -1 to move all to GPU.
    n_ctx = 4096        # Context window
    n_batch = 512       # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    _llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        temperature=temp,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    return _llm

################################################
# Create prompt object
def createPrompt():
    system_prompt = '''
        <|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant for technical advice and recommendations.<|eot_id|>
        Be concise. Do not provide unhelpful responses. If you do not know the answer, say you do not know.
        Respond to the user input based only on the following context:

        {context}
        <|eot_id|>
        '''
    user_prompt = '''
        <|start_header_id|>user<|end_header_id|>
        {input}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        '''
    _prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )
    return _prompt

# ################################################
# One-off QA chain (no memory)
def createQAChain(_llm, _vectorstore, _prompt):
    question_answer_chain = create_stuff_documents_chain(_llm, _prompt)
    _chat = create_retrieval_chain(
        _vectorstore.as_retriever(),
        question_answer_chain)    
    return _chat

# Instantiate Q&A chain ========================
vectorstore = loadVectorStore()
llm = loadLLM()
prompt = createPrompt()
chat = createQAChain(llm, vectorstore, prompt)

################################################
# see:
# - https://fastapi.tiangolo.com/tutorial/body/
# - https://stackoverflow.com/questions/64057445/fastapi-post-does-not-recognize-my-parameter
class Query(BaseModel):
    data: str

def getBotResponse(query):
  if validate_query_relevance(vectorstore, query):
      response = chat.invoke({"input": query}).get("result")
  else:
      response = '''Your input is too vague or is not related to content
      in the knowledgebase. Please rephrase.'''
  return response

app = FastAPI()

# Allow access from localhost
# see: https://fastapi.tiangolo.com/tutorial/cors/
'''
origins = ["*"]
'''
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
def _getBotResponse(query: Query):
    return getBotResponse(query.data)
