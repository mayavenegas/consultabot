#!/usr/bin/python3

# Import necessary libraries
import os
from typing import Any, Dict, List, Union
import numpy as np
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# for tracing
from langfuse.callback import CallbackHandler
lf_skey=os.environ['LANGFUSE_SECRET_KEY']
lf_pkey=os.environ['LANGFUSE_PUBLIC_KEY']
lf_host=os.environ['LANGFUSE_HOST']

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
    logging.info("Read vectorstore from disk...")

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
    logging.info(f"Doc search scores for: {query_text}:")
    for i in range(0,len(docs)):
        logging.info(f"\t{docs[i][1]}")
    # return True if all scores <= max_score
    return all(score <= max_score for (_, score) in docs)

################################################
# Load LLM from disk
def loadLLM():
    logging.info("Loading LLM from disk...")

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    max_tokens = 4096
    temp = 0            # stick to the facts
    n_gpu_layers = -1   # -1 to move all to GPU.
    n_ctx = 4096        # Context window
    n_batch = 512       # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    _llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        max_tokens=max_tokens,
        temperature=temp,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
    )
    return _llm

################################################
# Create prompt object
# see: https://github.com/meta-llama/llama3?tab=readme-ov-file#instruction-tuned-models
'''
        <|start_header_id|>system<|end_header_id|>        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
'''
def createPrompt():
    system_prompt = '''<|start_header_id|>system<|end_header_id|>
    
        You are a helpful AI assistant for technical advice and recommendations.<|eot_id|>
        Be concise. Do not provide unhelpful responses. If you do not know the answer, say you do not know.
        Respond to the user input based only on the following context:
        {context}<|eot_id|>'''
    user_prompt = '''<|start_header_id|>user<|end_header_id|>
    
        {input}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
        '''
    _prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    return _prompt

# ################################################
# One-off QA chain (no memory)
def createQAChain(_vectorstore, _llm, _prompt):
    question_answer_chain = create_stuff_documents_chain(_llm, _prompt)
    _chat = create_retrieval_chain(
                    _vectorstore.as_retriever(),
                    question_answer_chain
    )    
    return _chat

# ################################################
# Class to trace chain execution, inputs & outputs
INDENT=2
indent=0
class debugCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(" "*indent, f"LLM {serialized.get('name')} started")
        print(" "*indent, f"prompts: {prompts}")
        indent += INDENT

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        print(" "*indent, f"Chat {serialized.get('name')} started")
        print(" "*indent, f"messages: {messages}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(" "*indent, "LLM ended.")
        print(" "*indent, f"response: {response}")
        print(" "*indent, "="*40)
        indent -= INDENT

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        print(" "*indent, f"LLM error: {error}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        print(" "*indent, f"Chain {serialized.get('name')} started")
        print(" "*indent, f"inputs: {inputs}")
        indent += INDENT

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        print(" "*indent, f"Chain ended, outputs: {outputs}")
        print(" "*indent, "="*40)
        indent -= INDENT

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        print(" "*indent, " "*indent, f"Chain error: {error}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        print(" "*indent, f"Tool {serialized.get('name')} started")
        print(" "*indent, f"input_str: {input_str}")
        indent += INDENT

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        print(f"Tool ended, outputs: {output}")
        print("="*40)
        indent -= INDENT

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        print(" "*indent, f"Tool error: {error}")

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""

# Instantiate Q&A chain ========================
vectorstore = loadVectorStore()
llm = loadLLM()
prompt = createPrompt()
chat = createQAChain(vectorstore, llm, prompt)

# ------------------------------------------------------------------------------
# Define structure of query payload, see:
# - https://fastapi.tiangolo.com/tutorial/body/
# - https://stackoverflow.com/questions/64057445/fastapi-post-does-not-recognize-my-parameter
class Query(BaseModel):
    data: str

def getBotResponse(query):
    if validate_query_relevance(vectorstore, query):
        ''' == for tracing
        langfuse_handler = CallbackHandler(secret_key=lf_skey,public_key=lf_pkey,host=lf_host)
        raw_response = chat.invoke({"input": query}, config={"callbacks": [langfuse_handler]})
        '''
        logging.debug(f"query: {query}")
        raw_response = chat.invoke({"input": query}, config={"callbacks": [debugCallbackHandler()]})
        logging.debug(f"raw_response: {raw_response}")
        response = raw_response.get('answer')
    else:
        response = '''Your input is too vague or is not related to content
        in the knowledgebase. Please rephrase.'''
    return response

app = FastAPI()

# Allow access from localhost
# see: https://fastapi.tiangolo.com/tutorial/cors/
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
