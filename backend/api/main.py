from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sae_lens import SAE 
from threading import Thread 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer 

from api.constants import LAYER, MAX_TOKENS, MODEL_ID, SAE_ID, SAE_IDX, SAE_RELEASE, STEERING_COEFF
from api.models import ChatRequest, ChatResponse 

import os 
import torch 
import numpy as np 

models = {}

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
        models["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID)
        models["streamer"] = TextIteratorStreamer(models["tokenizer"])
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
    
    print("Loading LLM...")
    try:
        models["llm"] = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto")        
    except Exception as e:
        print(f"Failed to load LLM: {e}") 

    print("Loading SAE...")       
    try:
        models["sae"], _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)            
    except Exception as e:
        print(f"Failed to load SAE: {e}") 

    yield 
    
    print("Shutting down server...")
    models.clear()
    print("Models cleared from memory...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.get("/health")
async def health():
    return Response(status_code=200, content="ok")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:                 
        input_ids = generate_input_ids(models["tokenizer"], request.prompt, models["llm"].device)
        
        generation_kwargs = dict(input_ids=input_ids,
                            max_new_tokens=MAX_TOKENS,
                            streamer=models["streamer"],
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9)

        streamer, hook = hooked_generate(models["llm"], 
                                         models["tokenizer"], 
                                         generation_kwargs, 
                                         input_ids, 
                                         steering_vector=models["sae"].W_dec[SAE_IDX])
        

        try:
            for new_text in streamer:
                return ChatResponse(response=new_text)
        finally:
            hook.remove() 
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def generate_input_ids(tokenizer: AutoTokenizer, prompt: str, device: str) -> torch.Tensor:
    try:        
        return tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    except Exception as e:
        raise Exception(f"Failed to tokenize prompt: {e}")


def steering_hook(resid_post: tuple, 
                  steering_vector: torch.Tensor, 
                  steering_coeff: int, 
                  steering_on: bool = True) -> torch.Tensor:
    """
    Apply steering vector to residual stream during streaming generation
    """
    # Handle different output structures
    if isinstance(resid_post, tuple):
        # If it's a tuple, we want the hidden states
        hidden_states = resid_post[0]
    else:
        # If it's not a tuple, it's likely already the hidden states
        hidden_states = resid_post

    if steering_on:
        # Apply steering vector to the current token
        steering = steering_vector.to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states + steering_coeff * steering

    # Return in the same format as received
    if isinstance(resid_post, tuple):
        return (hidden_states,) + resid_post[1:]
    return hidden_states

def hooked_generate(model: AutoModelForCausalLM, 
                    tokenizer: AutoTokenizer, 
                    generation_kwargs: dict, 
                    input_ids: torch.Tensor, 
                    steering_vector: torch.Tensor,                     
                    layer: int = LAYER, 
                    steering_coeff: int = STEERING_COEFF, 
                    seed: int = 42, 
                    **kwargs) -> str:
    """
    Generate text with steering vector intervention
    """
    if seed:
        torch.manual_seed(seed)

    def hook_fn(module, inputs, outputs):
        return steering_hook(outputs, steering_vector, steering_coeff)

    # Register the hook
    hook = model.model.layers[layer].register_forward_hook(hook_fn)

    try:    
        streamer = generation_kwargs["streamer"]                
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer, hook

    except Exception as e:
        hook.remove()
        raise Exception(f"Failed to generate text: {e}")