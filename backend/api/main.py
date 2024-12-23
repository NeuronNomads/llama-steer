from contextlib import asynccontextmanager
from fastapi import FastAPI, Response 
from fastapi.middleware.cors import CORSMiddleware
from sae_lens import SAE 
from threading import Thread 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer 

import torch 
import numpy as np 


@asynccontextmanager 
async def lifespan(app: FastAPI): 
    print("Starting llama server")
    yield 
    print("Shutting down server")

app = FastAPI(lifespan=lifespan)
origins = ["*"]
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.get("/health")
async def health():
    return Response(status_code=200, content="ok")

@app.post("/chat")
async def chat():
    pass 






