# in app/main.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import requests

# --- Cloud and Local Imports ---
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from app.vector_store import VectorStore, FAISSVectorStore, VertexAIVectorStore
from ingestion.text_processor import clean_gutenberg_text, chunk_text_by_paragraph

# --- Load Environment and Initialize ---
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DEV_ENVIRONMENT = os.getenv("DEV_ENVIRONMENT", "local")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

vertexai.init(project=PROJECT_ID, location=LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
generative_model = GenerativeModel("gemini-2.5-flash")

# --- Global Vector Store variable ---
vector_store: VectorStore = None
local_text_map: dict = {}

# --- FastAPI App ---
app = FastAPI(title="SCRIBE v3 API")

@app.on_event("startup")
def startup_event():
    """Initializes the vector store based on the environment."""
    global vector_store
    global local_text_map
    
    if DEV_ENVIRONMENT == "local":
        # In local mode, we load the FAISS index from disk.
        BOOK_ID = "2680" # The book we've indexed locally
        faiss_store = FAISSVectorStore(
            index_path=f"app/local_index_{BOOK_ID}.faiss",
            metadata_path=f"app/local_metadata_{BOOK_ID}.pkl"
        )
        faiss_store.load()
        vector_store = faiss_store
        local_text_map = faiss_store.text_map # Store the text map for retrieval
    
    elif DEV_ENVIRONMENT == "cloud":
        # In cloud mode, we connect to the live Vertex AI endpoint.
        ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")
        DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID")
        vector_store = VertexAIVectorStore(ENDPOINT_ID, DEPLOYED_INDEX_ID)
    
    else:
        raise ValueError(f"Unknown environment: {DEV_ENVIRONMENT}")


def get_text_for_chunks(chunk_ids: list[str]) -> str:
    """Retrieves the text for chunk IDs, using the appropriate method for the environment."""
    if DEV_ENVIRONMENT == "local":
        # Local: Look up text in the loaded metadata map
        return "\n---\n".join([local_text_map.get(cid, "") for cid in chunk_ids])
    
    else: # Cloud
        # Cloud: Re-download and process the book (inefficient MVP method)
        books = {}
        # ... (The rest of this cloud-based retrieval function is the same as before)
        # ... (This can be copied from the previous version of main.py if needed)
        return "Cloud text retrieval not fully implemented in this snippet."


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: str

@app.post("/query", response_model=QueryResponse)
def query_index(request: QueryRequest):
    """
    Accepts a user's question, finds relevant chunks, and returns a grounded answer.
    """
    global vector_store
    print(f"Received query in '{DEV_ENVIRONMENT}' mode: {request.question}")

    # 1. Embed the user's question
    query_embedding = embedding_model.get_embeddings([request.question])[0].values
    
    # 2. Find relevant document chunks (using our abstraction)
    found_chunk_ids = vector_store.find_neighbors(query_embedding, num_neighbors=3)
    print(f"Found neighbors: {found_chunk_ids}")

    # 3. Retrieve the actual text for those chunks
    context_text = get_text_for_chunks(found_chunk_ids)
    
    # 4. Engineer a prompt and generate a final answer
    prompt = f"""
    You are SCRIBE, a scholarly assistant. Answer the user's question based *only* on the provided context.

    CONTEXT:
    {context_text}

    QUESTION:
    {request.question}

    ANSWER:
    """
    generation_response = generative_model.generate_content(prompt)
    answer = generation_response.text
    
    return {"question": request.question, "answer": answer, "context": context_text}

@app.get("/")
def read_root():
    """ A simple health check endpoint. """
    return {"status": "ok"}