from contextlib import asynccontextmanager
from fastapi import FastAPI, Response 
from fastapi.middleware.cors import CORSMiddleware
from sae_lens import SAE 
from threading import Thread 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer 

from api.constants import MODEL_ID, SAE_ID, SAE_RELEASE

import os 
import torch 
import numpy as np 


@asynccontextmanager 
async def lifespan(app: FastAPI): 
    """
    Utilize the lifespan to load in tokenizer, LLM and SAE
    """

    print("Determine device...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")


    print("Loading tokenizer...")    
    try: 
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        streamer = TextIteratorStreamer(tokenizer)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
    
    print("Loading LLM...")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_2")
        model.to(device)
    except Exception as e:
        print(f"Failed to load LLM: {e}") 

    print("Loading SAE...")       
    try:
        sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)            
    except Exception as e:
        print(f"Failed to load SAE: {e}") 

    yield 
    
    print("Shutting down server...")

    del sae 
    del model 
    del streamer
    del tokenizer 

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.get("/health")
async def health():
    return Response(status_code=200, content="ok")

@app.post("/chat")
async def chat():
    pass 






