import os
import argparse
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel

import faiss
import numpy as np
import pickle

from vertexai.language_models import TextEmbeddingModel
from prompts import CORE_PERSONA_PROMPT

VERSE_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'app', 'local_verses_genesis.faiss')
VERSE_METADATA_PATH = os.path.join(os.path.dirname(__file__), 'app', 'local_verses_genesis.pkl')
PERICOPE_METADATA_PATH = os.path.join(os.path.dirname(__file__), 'app', 'local_pericopes_genesis.pkl')

class ConversationalAgent:
    def __init__(self, model_name):
        self.model = GenerativeModel(model_name)
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.history = []
        self.verse_index = None
        self.verse_metadata = None
        self.pericope_metadata_store = None
        self._load_index()

    def _load_index(self):
        try:
            print("--- Loading and Preparing Index and Metadata ---")
            # Load the verse index
            self.verse_index = faiss.read_index(VERSE_INDEX_PATH)
            # Load the verse metadata
            with open(VERSE_METADATA_PATH, 'rb') as f:
                self.verse_metadata = pickle.load(f)
            # Load the pericope metadata store
            with open(PERICOPE_METADATA_PATH, 'rb') as f:
                self.pericope_metadata_store = pickle.load(f)

            print(f"  -> Verse index loaded with {self.verse_index.ntotal} vectors.")
            print(f"  -> Verse metadata loaded for {len(self.verse_metadata)} verses.")
            print(f"  -> Pericope metadata store loaded for {len(self.pericope_metadata_store)} pericopes.")

        except Exception as e:
            print(f"Error loading local index: {e}")

    def _rewrite_query(self, query):
        """Rewrites a potentially ambiguous query into a self-contained one."""
        if not self.history:
            return query

        history_string = "\n".join([f"{turn['role']}: {turn['content']}" for turn in self.history])
        
        prompt = f"""
        Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.

        Conversation History:
        {history_string}

        Follow-up Question: {query}

        Standalone Question:
        """
        
        response = self.model.generate_content(prompt)
        rewritten_query = response.text.strip()
        print(f"  -> Rewritten Query: {rewritten_query}")
        return rewritten_query

    def _search(self, query):
        """Performs a small-to-big vector search."""
        print(f"  -> Searching for: '{query}'")
        query_embedding = self.embedding_model.get_embeddings([query])[0].values
        k = 1 # Find the single most relevant verse
        distances, indices = self.verse_index.search(np.array([query_embedding], dtype=np.float32), k)
        
        # Get the parent ID of the most relevant verse
        if not indices.any():
            print("  -> No relevant verses found.")
            return []

        most_relevant_verse_index = indices[0][0]
        most_relevant_verse_metadata = self.verse_metadata[most_relevant_verse_index]
        parent_id = most_relevant_verse_metadata.get('parent_chunk_id')

        if not parent_id:
            print("  -> Could not find parent ID for the most relevant verse.")
            return []

        # Retrieve the full parent chunk
        print(f"  -> Found parent ID: {parent_id}")
        retrieved_pericope = self.pericope_metadata_store.get(parent_id)
        
        if not retrieved_pericope:
            print(f"  -> Could not retrieve pericope for ID: {parent_id}")
            return []

        print(f"  -> Retrieved 1 parent pericope based on verse search.")
        return [retrieved_pericope]

    def _generate_answer(self, query, context_chunks):
        """Generates a final answer based on the retrieved context."""
        context_string = ""
        for chunk in context_chunks:
            context_string += f"Reference: {chunk.get('canonical_reference')}\n"
            context_string += f"Text: {chunk.get('text')}\n\n"

        prompt = f"""
        {CORE_PERSONA_PROMPT}

        Answer the following question using only the provided context. Be concise and direct.

        Context:
        ---
        {context_string}
        ---

        Question: {query}

        Answer:
        """
        
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def chat(self, query):
        """Main chat function to handle a user query."""
        print(f"\nUser: {query}")
        self.history.append({"role": "user", "content": query})

        rewritten_query = self._rewrite_query(query)
        retrieved_chunks = self._search(rewritten_query)
        answer = self._generate_answer(rewritten_query, retrieved_chunks)

        print(f"\nSCRIBE: {answer}")
        self.history.append({"role": "model", "content": answer})

def main():
    """Main function to run the conversational chat."""
    parser = argparse.ArgumentParser(description="Have a conversational chat with SCRIBE.")
    parser.add_argument("query", type=str, help="Your question for SCRIBE.")
    args = parser.parse_args()

    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    model_name = os.getenv("VERTEX_GENERATIVE_MODEL")

    if not all([project_id, location, model_name]):
        print("Error: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and VERTEX_GENERATIVE_MODEL must be set in .env")
        return

    vertexai.init(project=project_id, location=location)
    
    agent = ConversationalAgent(model_name)
    agent.chat(args.query)

if __name__ == "__main__":
    main()

