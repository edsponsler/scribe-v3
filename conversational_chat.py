import os
import argparse
import json
import re
import glob
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel

import faiss
import numpy as np
import pickle

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from prompts import CORE_PERSONA_PROMPT

class ConversationalAgent:
    def __init__(self, model_name, source="genesis"):
        self.model = GenerativeModel(model_name)
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.history = []
        self.source = source
        self.search_indexes = {}
        self.search_metadatas = {}
        self.retrieval_stores = {}
        self.book_metadata = {}
        self._load_index()

    def _load_index(self):
        """Loads all the necessary data artifacts for a given source."""
        try:
            print(f"--- Loading and Preparing Index and Metadata for source: {self.source} ---")
            app_dir = os.path.join(os.path.dirname(__file__), 'app')

            if self.source == "genesis":
                print("  -> Loading unified data model for Genesis...")
                self.search_indexes['summaries'] = faiss.read_index(os.path.join(app_dir, 'local_summaries_genesis.faiss'))
                with open(os.path.join(app_dir, 'local_summaries_genesis.pkl'), 'rb') as f:
                    self.search_metadatas['summaries'] = pickle.load(f)
                
                self.search_indexes['verses'] = faiss.read_index(os.path.join(app_dir, 'local_verses_genesis.faiss'))
                with open(os.path.join(app_dir, 'local_verses_genesis.pkl'), 'rb') as f:
                    self.search_metadatas['verses'] = pickle.load(f)
                
                with open(os.path.join(app_dir, 'local_chapters_genesis.pkl'), 'rb') as f:
                    self.retrieval_stores['chapters'] = pickle.load(f)
                with open(os.path.join(app_dir, 'local_pericopes_genesis.pkl'), 'rb') as f:
                    self.retrieval_stores['pericopes'] = pickle.load(f)

                with open(os.path.join(app_dir, 'local_book_metadata_genesis.json'), 'r') as f:
                    self.book_metadata = json.load(f)

            elif self.source == "gutenberg":
                # This logic can be expanded later if needed
                pass
            else:
                raise ValueError(f"Unknown source: {self.source}")

            print(f"  -> All data for source '{self.source}' loaded successfully.")

        except Exception as e:
            print(f"Error loading local index: {e}")

    def _rewrite_query(self, query):
        """Rewrites a potentially ambiguous query into a self-contained one."""
        if not self.history:
            return query
        # Simplified for now
        return query

    def _search_genesis(self, query_embedding):
        """Performs a parallel search for relevant summaries and verses."""
        print("  -> Executing parallel search for Genesis source...")

        # --- Search 1: Verses (for specific facts) ---
        verse_index = self.search_indexes['verses']
        verse_metadatas = self.search_metadatas['verses']
        v_distances, v_indices = verse_index.search(np.array([query_embedding], dtype=np.float32), 7)
        top_verses = [verse_metadatas[i] for i in v_indices[0]] if v_indices.any() else []
        print(f"    -> Found {len(top_verses)} relevant verses.")

        # --- Search 2: Summaries (for high-level context) ---
        summary_index = self.search_indexes['summaries']
        summary_metadatas = self.search_metadatas['summaries']
        s_distances, s_indices = summary_index.search(np.array([query_embedding], dtype=np.float32), 3)
        top_summaries = [summary_metadatas[i] for i in s_indices[0]] if s_indices.any() else []
        print(f"    -> Found {len(top_summaries)} relevant summaries.")

        return top_summaries, top_verses

    def _search(self, query):
        """Orchestrates the search by dispatching to the correct method based on source."""
        print(f"  -> Searching for: '{query}'")
        query_embedding = self.embedding_model.get_embeddings([query])[0].values
        
        if self.source == "genesis":
            return self._search_genesis(query_embedding)
        else:
            return [], []

    def _generate_answer(self, query, top_summaries, top_verses):
        """Generates a final answer based on the retrieved context."""
        if not top_summaries and not top_verses:
            return "I could not find relevant passages to answer your question."

        # --- Construct Context Block ---
        summaries_context = ""
        if top_summaries:
            for meta in top_summaries:
                ref = meta.get('canonical_reference', 'Unknown Reference')
                text = meta.get('text', 'No summary available.')
                level = meta.get('chunk_level', 'chapter')
                
                label = f"Pericope {ref}" if level == 'pericope' else ref
                summaries_context += f"{label}: {text}\n"
        
        verses_context = ""
        if top_verses:
            try:
                top_verses.sort(key=lambda x: list(map(int, re.findall(r'\d+', x['canonical_reference'])))) 
            except (ValueError, TypeError): pass
            
            for meta in top_verses:
                verses_context += f"({meta['canonical_reference']}) {meta['text']}\n"

        # --- Create Final Prompt ---
        prompt = f"""{CORE_PERSONA_PROMPT}

Use the following retrieved context to answer the user's question.

--- Summaries ---
{summaries_context or 'No relevant summaries found.'}
--- End Summaries ---

--- Relevant Verses ---
{verses_context or 'No relevant verses found.'}
--- End Relevant Verses ---

Question: {query}

Answer:"""
        
        response = self.model.generate_content(prompt)
        answer = response.text.strip()
        
        final_reference = f"Source: *{self.book_metadata.get('title', 'Genesis')}*"
        return f"{answer}\n\n{final_reference}"

    def chat(self, query):
        """Main chat function to handle a user query."""
        self.history.append({"role": "user", "content": query})
        rewritten_query = self._rewrite_query(query)
        top_summaries, top_verses = self._search(rewritten_query)
        answer = self._generate_answer(rewritten_query, top_summaries, top_verses)
        self.history.append({"role": "model", "content": answer})
        return answer

if __name__ == "__main__":
    # CLI testing logic
    pass