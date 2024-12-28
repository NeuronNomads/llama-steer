from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect 
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer 

from api.constants import LAYER, MAX_TOKENS, MODEL_ID, STEERING_COEFF
 
import os 
import time 
import torch 
import numpy as np 

models = {}
hf_token = os.environ.get("HF_TOKEN")
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
    
    WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")
    print("this is weights_path", WEIGHTS_PATH)

    print("Loading tokenizer...")    
    try: 
        models["tokenizer"] = AutoTokenizer.from_pretrained("gpt2")
        print("Succesfully loaded tokenizer...")

        models["streamer"] = TextIteratorStreamer(models["tokenizer"])
        print("Succesfully loaded streamer...", flush=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
    
    print("Loading LLM...", flush=True)
    try:
        models["llm"] = AutoModelForCausalLM.from_pretrained("gpt2",
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto",                                                     
)   
        print("Succesfully loaded steering LLM...", flush=True)
    except Exception as e:
        print(f"Failed to load LLM: {e}", flush=True) 

    print("Loading Steering Vector...")       
    try:        
        models["steering_vector"] = torch.load("api/steering_vectors/packers_steering_vector.pt", map_location=device)         
        print("Succesfully loaded steering vector...")
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

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    try:
        prompt = await websocket.receive_text()
        
        try:
            input_ids = generate_input_ids(models["tokenizer"], prompt, models["llm"].device)
            
            generation_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=MAX_TOKENS,
                streamer=models["streamer"],
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )

            streamer, hook, thread = await hooked_generate(models["llm"],
                                           models["tokenizer"],
                                           generation_kwargs,
                                           input_ids,
                                           steering_vector=models["steering_vector"])

            
            # Stream the tokens as they're generated
            st = time.time()                
            for text in streamer:                                        
                await websocket.send_text(text)
                et = time.time()
                print(f"Time taken to send text {text} is {et - st} seconds")
                st = time.time()                                

        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")
            raise e
        

    except WebSocketDisconnect:
        print("Client disconnected")
    
    finally:
        await websocket.close()
        if hook:
            print("Removing hook...")
            hook.remove()
        if thread and thread.is_alive():
            print("Removing thread...")
            thread.join(timeout=1.0)

def generate_input_ids(tokenizer: AutoTokenizer, prompt: str, device: str) -> torch.Tensor:
    try:        
        return tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    except Exception as e:
        raise Exception(f"Failed to tokenize prompt: {e}")


def steering_hook(resid_pre: tuple, 
                  steering_vector: torch.Tensor, 
                  steering_coeff: int, 
                  steering_on: bool = True) -> torch.Tensor:
    """
    Apply steering vector to residual stream during streaming generation
    """
    """
    Apply steering vector to residual stream
    """
    # If output is a tuple, take the first element
    if isinstance(resid_pre, tuple):        
        resid_pre = resid_pre[0]        
        
    if resid_pre.shape[1] == 1:  # Skip if only one token        
        return
    
    position = resid_pre.shape[1]
    if steering_on:
        # Apply steering vector to all positions except the last one
        resid_pre[:, :position-1, :] += steering_coeff * steering_vector.to(models["llm"].device)

async def hooked_generate(model: AutoModelForCausalLM,
                         tokenizer: AutoTokenizer,
                         generation_kwargs: dict,
                         input_ids: torch.Tensor,
                         steering_vector: torch.Tensor,
                         layer: int = LAYER,
                         steering_coeff: int = STEERING_COEFF,
                         seed: int = 42,
                         **kwargs) -> tuple:
    """
    Generate text with steering vector intervention
    """
    if seed:
        torch.manual_seed(seed)

    def hook_fn(module, inputs):
        steering_hook(inputs[0], steering_vector, steering_coeff)
        return inputs

    # Register the hook    
    hook = model.transformer.h[layer].register_forward_pre_hook(hook_fn)    

    try:        
        streamer = generation_kwargs["streamer"]                                
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()        
        
        return streamer, hook, thread 

    except Exception as e:
        hook.remove()
        raise Exception(f"Failed to generate text: {e}")
        
