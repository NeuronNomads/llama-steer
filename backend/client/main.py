import asyncio
import sys 
import websockets

DEFAULT_URL = "ws://127.0.0.1:8000/chat"

async def main():
    async with websockets.connect(
        DEFAULT_URL, 
        close_timeout=60,
        ping_interval=20, 
        ping_timeout=20
    ) as websocket:
        # Send the prompt as plain text since server expects plain text
        prompt = "Tell me about Einstein"
        await websocket.send(prompt)
        
        try:
            # Accumulate the full response
            full_response = ""
            
            while True:
                try:
                    # Increased timeout for LLM generation
                    response = await asyncio.wait_for(websocket.recv(), timeout=60)
                    print(response, end="", flush=True)
                    full_response += response
                    
                except asyncio.TimeoutError:
                    # If we timeout, assume generation is complete
                    break
                except websockets.exceptions.ConnectionClosed:
                    # Handle normal websocket closure
                    print("\nConnection closed normally")
                    break
        
        except Exception as e:
            print(f"\nError occurred: {e}")
        
        print("\n\nFull response:", full_response)

if __name__ == "__main__":
    asyncio.run(main())