import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sys

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversational_chat import ConversationalAgent

# --- Load Environment and Initialize --
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GENERATIVE_MODEL_NAME = os.getenv("VERTEX_GENERATIVE_MODEL", "gemini-1.5-flash")

# --- FastAPI App ---
app = FastAPI(title="SCRIBE v3 API")

# --- Global Agent Dictionary ---
# We will store our pre-loaded agents here
agents = {}

@app.on_event("startup")
def startup_event():
    """
    Initializes and loads the conversational agents for each source at startup.
    """
    print("--- Server is starting up. Loading conversational agents... ---")
    global agents
    
    # Initialize Genesis Agent
    print("  -> Initializing agent for source: genesis")
    agents["genesis"] = ConversationalAgent(GENERATIVE_MODEL_NAME, source="genesis")
    
    # Initialize Gutenberg Agent
    print("  -> Initializing agent for source: gutenberg")
    agents["gutenberg"] = ConversationalAgent(GENERATIVE_MODEL_NAME, source="gutenberg")
    
    print("--- All agents loaded and ready. ---")


# --- API Models ---
class ChatRequest(BaseModel):
    message: str
    source: str # e.g., "genesis" or "gutenberg"

class ChatResponse(BaseModel):
    reply: str

# --- API Endpoints ---
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_root():
    """ Serves the main chat interface. """
    return FileResponse('app/static/index.html')

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Accepts a user's message and a source, and returns the agent's response.
    """
    global agents
    print(f"Received message for source '{request.source}': {request.message}")
    
    agent = agents.get(request.source)
    
    if not agent:
        return {"reply": f"Error: No agent found for source '{request.source}'. Available sources are: {list(agents.keys())}"}
        
    reply = agent.chat(request.message)
    
    return {"reply": reply}