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
        print(f"--- Loading and Preparing Index and Metadata for source: {self.source} ---")
        app_dir = os.path.join(os.path.dirname(__file__), 'app')

        try:
            if self.source == "genesis":
                self._load_genesis_data(app_dir)
            elif self.source == "gutenberg":
                self._load_gutenberg_data(app_dir)
            else:
                raise ValueError(f"Unknown source: {self.source}")

            print(f"  -> All data for source '{self.source}' loaded successfully.")

        except Exception as e:
            print(f"Error loading local index: {e}")

    def _load_genesis_data(self, app_dir):
        """Loads the data artifacts for the 'genesis' source."""
        print("  -> Loading unified data model for Genesis...")
        self.search_indexes['summaries'] = faiss.read_index(os.path.join(app_dir, 'local_summaries_genesis.faiss'))
        with open(os.path.join(app_dir, 'local_summaries_genesis.pkl'), 'rb') as f:
            self.search_metadatas['summaries'] = pickle.load(f)
        
        self.search_indexes['details'] = faiss.read_index(os.path.join(app_dir, 'local_verses_genesis.faiss'))
        with open(os.path.join(app_dir, 'local_verses_genesis.pkl'), 'rb') as f:
            self.search_metadatas['details'] = pickle.load(f)
        
        with open(os.path.join(app_dir, 'local_chapters_genesis.pkl'), 'rb') as f:
            self.retrieval_stores['chapters'] = pickle.load(f)
        with open(os.path.join(app_dir, 'local_pericopes_genesis.pkl'), 'rb') as f:
            self.retrieval_stores['pericopes'] = pickle.load(f)

        with open(os.path.join(app_dir, 'local_book_metadata_genesis.json'), 'r') as f:
            self.book_metadata = json.load(f)

    def _load_gutenberg_data(self, app_dir):
        """Loads and merges data artifacts for the 'gutenberg' source."""
        print("  -> Loading and merging data models for Gutenberg...")
        book_id = "2680" # Hardcoded for now

        # --- Load and merge paragraph (detail) data ---
        detail_metadatas = []
        for meta_path in glob.glob(os.path.join(app_dir, f'local_*_search_{book_id}.pkl')):
            if os.path.getsize(meta_path) > 0:
                with open(meta_path, 'rb') as f:
                    detail_metadatas.extend(pickle.load(f))
        
        faiss_files = glob.glob(os.path.join(app_dir, f'local_*_search_{book_id}.faiss'))
        merged_detail_index = faiss.IndexFlatL2(768) # Assuming embedding dim is 768
        for f_path in faiss_files:
            if os.path.getsize(f_path) > 0:
                index = faiss.read_index(f_path)
                merged_detail_index.add(index.reconstruct_n(0, index.ntotal))

        self.search_indexes['details'] = merged_detail_index
        self.search_metadatas['details'] = detail_metadatas

        # --- Load and merge summary data ---
        summary_metadatas = []
        for meta_path in glob.glob(os.path.join(app_dir, f'local_*_summaries_{book_id}.pkl')):
            if os.path.getsize(meta_path) > 0:
                with open(meta_path, 'rb') as f:
                    summary_metadatas.extend(pickle.load(f))

        faiss_files = glob.glob(os.path.join(app_dir, f'local_*_summaries_{book_id}.faiss'))
        merged_summary_index = faiss.IndexFlatL2(768)
        for f_path in faiss_files:
            if os.path.getsize(f_path) > 0:
                index = faiss.read_index(f_path)
                merged_summary_index.add(index.reconstruct_n(0, index.ntotal))

        self.search_indexes['summaries'] = merged_summary_index
        self.search_metadatas['summaries'] = summary_metadatas

        # --- Load and merge retrieval stores ---
        self.retrieval_stores['sections'] = {}
        for store_path in glob.glob(os.path.join(app_dir, f'local_*_retrieval_{book_id}.pkl')):
            if os.path.getsize(store_path) > 0:
                with open(store_path, 'rb') as f:
                    self.retrieval_stores['sections'].update(pickle.load(f))

        with open(os.path.join(app_dir, f'local_book_metadata_{book_id}.json'), 'r') as f:
            self.book_metadata = json.load(f)

    def _search(self, query):
        """Orchestrates the search by dispatching to the correct method based on source."""
        print(f"  -> Searching for: '{query}'")
        query_embedding = self.embedding_model.get_embeddings([query])[0].values
        
        print("  -> Executing parallel search...")
        # --- Search 1: Details (for specific facts) ---
        detail_index = self.search_indexes['details']
        detail_metadatas = self.search_metadatas['details']
        d_distances, d_indices = detail_index.search(np.array([query_embedding], dtype=np.float32), 7)
        top_details = [detail_metadatas[i] for i in d_indices[0]] if d_indices.size > 0 else []
        print(f"    -> Found {len(top_details)} relevant details.")

        # --- Search 2: Summaries (for high-level context) ---
        summary_index = self.search_indexes['summaries']
        summary_metadatas = self.search_metadatas['summaries']
        s_distances, s_indices = summary_index.search(np.array([query_embedding], dtype=np.float32), 3)
        top_summaries = [summary_metadatas[i] for i in s_indices[0]] if s_indices.size > 0 else []
        print(f"    -> Found {len(top_summaries)} relevant summaries.")

        return top_summaries, top_details

    def _generate_answer(self, query, top_summaries, top_details):
        """Generates a final answer based on the retrieved context."""
        if not top_summaries and not top_details:
            return "I could not find relevant passages to answer your question."

        # --- Construct Context Block ---
        summaries_context = ""
        if top_summaries:
            for meta in top_summaries:
                # For summaries, we retrieve the full parent text for better context
                parent_id = meta.get('parent_chunk_ids', [None])[0]
                if parent_id and self.source == 'gutenberg':
                    full_text = self.retrieval_stores['sections'][parent_id].get('text', '')
                    ref = self.retrieval_stores['sections'][parent_id].get('canonical_reference', 'Unknown Reference')
                    summaries_context += f"CONTEXT FROM SECTION: {ref}\n{full_text}\n---\n"
                else: # Fallback for Genesis or if parent lookup fails
                    ref = meta.get('canonical_reference', 'Unknown Reference')
                    text = meta.get('text', 'No summary available.')
                    summaries_context += f"{ref}: {text}\n"
        
        details_context = ""
        if top_details:
            # Sort details by canonical reference if possible
            try:
                top_details.sort(key=lambda x: list(map(int, re.findall(r'\d+', x['canonical_reference'])))) 
            except (ValueError, TypeError): pass
            
            for meta in top_details:
                details_context += f"({meta['canonical_reference']}) {meta['text']}\n"

        # --- Determine context labels based on source ---
        summary_label = "High-Level Summaries"
        detail_label = "Relevant Verses" if self.source == 'genesis' else "Relevant Paragraphs"

        # --- Create Final Prompt ---
        prompt = f"""{CORE_PERSONA_PROMPT}

Use the following retrieved context to answer the user's question. The context is divided into high-level summaries (which provide broad understanding) and specific paragraphs/verses (which provide fine-grained details).

--- {summary_label} ---
{summaries_context or 'No relevant summaries found.'}
--- End {summary_label} ---

--- {detail_label} ---
{details_context or 'No relevant details found.'}
--- End {detail_label} ---

Question: {query}

Answer:"""
        
        response = self.model.generate_content(prompt)
        answer = response.text.strip()
        
        final_reference = f"Source: *{self.book_metadata.get('title', self.source.title())}*"
        return f"{answer}\n\n{final_reference}"

    def chat(self, query):
        """Main chat function to handle a user query."""
        self.history.append({"role": "user", "content": query})
        top_summaries, top_details = self._search(query)
        answer = self._generate_answer(query, top_summaries, top_details)
        self.history.append({"role": "model", "content": answer})
        return answer

if __name__ == "__main__":
    # CLI testing logic
    pass
