import os
import argparse
import json
import re
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
        self.glossary = {}
        self.book_metadata = {}
        self._load_index()

    def _load_index(self):
        try:
            print(f"--- Loading and Preparing Index and Metadata for source: {self.source} ---")
            
            if self.source == "genesis":
                # For Genesis, we still use the simpler, single-index structure
                self.search_indexes['main'] = faiss.read_index(os.path.join(os.path.dirname(__file__), 'app', 'local_verses_genesis.faiss'))
                with open(os.path.join(os.path.dirname(__file__), 'app', 'local_verses_genesis.pkl'), 'rb') as f:
                    self.search_metadatas['main'] = pickle.load(f)
                with open(os.path.join(os.path.dirname(__file__), 'app', 'local_pericopes_genesis.pkl'), 'rb') as f:
                    self.retrieval_stores['main'] = pickle.load(f)

            elif self.source == "gutenberg":
                BOOK_ID = "2680"
                # Load main text files
                self.search_indexes['main_text'] = faiss.read_index(os.path.join(os.path.dirname(__file__), 'app', f'local_main_text_search_{BOOK_ID}.faiss'))
                with open(os.path.join(os.path.dirname(__file__), 'app', f'local_main_text_search_{BOOK_ID}.pkl'), 'rb') as f:
                    self.search_metadatas['main_text'] = pickle.load(f)
                with open(os.path.join(os.path.dirname(__file__), 'app', f'local_main_text_retrieval_{BOOK_ID}.pkl'), 'rb') as f:
                    self.retrieval_stores['main_text'] = pickle.load(f)
                # Load appendix files
                self.search_indexes['appendix'] = faiss.read_index(os.path.join(os.path.dirname(__file__), 'app', f'local_appendix_search_{BOOK_ID}.faiss'))
                with open(os.path.join(os.path.dirname(__file__), 'app', f'local_appendix_search_{BOOK_ID}.pkl'), 'rb') as f:
                    self.search_metadatas['appendix'] = pickle.load(f)
                with open(os.path.join(os.path.dirname(__file__), 'app', f'local_appendix_retrieval_{BOOK_ID}.pkl'), 'rb') as f:
                    self.retrieval_stores['appendix'] = pickle.load(f)
                # Load glossary and book metadata
                with open(os.path.join(os.path.dirname(__file__), 'app', f'local_glossary_{BOOK_ID}.pkl'), 'rb') as f:
                    self.glossary = pickle.load(f)
                with open(os.path.join(os.path.dirname(__file__), 'app', f'local_book_metadata_{BOOK_ID}.json'), 'r') as f:
                    self.book_metadata = json.load(f)
            else:
                raise ValueError(f"Unknown source: {self.source}")

            print(f"  -> All data for source '{self.source}' loaded successfully.")

        except Exception as e:
            print(f"Error loading local index: {e}")

    def _rewrite_query(self, query):
        """Rewrites a potentially ambiguous query into a self-contained one."""
        if not self.history:
            return query
        history_string = "\n".join([f"{turn['role']}: {turn['content']}" for turn in self.history])
        prompt = f'''Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.\n\nConversation History:\n{history_string}\n\nFollow-up Question: {query}\n\nStandalone Question:'''
        response = self.model.generate_content(prompt)
        rewritten_query = response.text.strip()
        print(f"  -> Rewritten Query: {rewritten_query}")
        return rewritten_query

    def _search(self, query):
        """Performs a unified vector search across all available indexes for the source."""
        print(f"  -> Searching for: '{query}'")
        query_embedding = self.embedding_model.get_embeddings([query])[0].values
        k = 1

        top_results = []
        for section_name, search_index in self.search_indexes.items():
            distances, indices = search_index.search(np.array([query_embedding], dtype=np.float32), k)
            if indices.any():
                top_results.append({
                    "distance": distances[0][0],
                    "index": indices[0][0],
                    "section": section_name
                })
        
        if not top_results:
            print("  -> No relevant chunks found in any index.")
            return None, None

        # Find the best result across all sections
        best_result = min(top_results, key=lambda x: x['distance'])
        print(f"  -> Top result found in section: '{best_result['section']}'")

        # Retrieve the metadata for the best result
        top_section = best_result['section']
        top_index = best_result['index']
        most_relevant_chunk_metadata = self.search_metadatas[top_section][top_index]
        parent_id = most_relevant_chunk_metadata.get('parent_chunk_id')

        if not parent_id:
            print("  -> Could not find parent ID for the most relevant chunk.")
            return None, None

        # Retrieve the full parent chunk from the correct retrieval store
        print(f"  -> Found parent ID: {parent_id}")
        retrieved_parent = self.retrieval_stores[top_section].get(parent_id)
        
        if not retrieved_parent:
            print(f"  -> Could not retrieve parent chunk for ID: {parent_id}")
            return None, None

        print(f"  -> Retrieved 1 parent chunk based on unified search.")
        return retrieved_parent, most_relevant_chunk_metadata

    def _generate_answer(self, query, parent_chunk, child_chunk):
        """Generates a final answer based on the retrieved context, footnotes, and glossary."""
        if not parent_chunk:
            return "I could not find a relevant passage to answer your question."

        # --- Augment the context ---
        parent_text = parent_chunk.get('text', '')
        footnote_map = parent_chunk.get('footnote_map', {})
        
        # Find relevant footnotes
        referenced_footnotes = {}
        for marker, text in footnote_map.items():
            if marker in parent_text:
                referenced_footnotes[marker] = text

        # Find relevant glossary terms
        defined_terms = {}
        if self.glossary:
            for term, definition in self.glossary.items():
                # Use word boundaries to avoid matching parts of words
                if re.search(r'\b' + re.escape(term) + r'\b', parent_text, re.IGNORECASE):
                    defined_terms[term] = definition

        # --- Build the prompt ---
        context_string = f"Reference: {parent_chunk.get('canonical_reference', 'N/A')}\n"
        context_string += f"Text: {parent_text}\n\n"

        if referenced_footnotes:
            footnote_string = "\n".join([f"{marker}: {text}" for marker, text in referenced_footnotes.items()])
            context_string += f"RELEVANT FOOTNOTES:\n---\n{footnote_string}\n---\n\n"

        if defined_terms:
            glossary_string = "\n".join([f"- {term}: {definition}" for term, definition in defined_terms.items()])
            context_string += f"RELEVANT GLOSSARY TERMS:\n---\n{glossary_string}\n---\n\n"

        prompt = f'''{CORE_PERSONA_PROMPT}\n\nAnswer the following question using only the provided context. Be concise and direct.\n\nContext:\n---\n{context_string}---\n\nQuestion: {query}\n\nAnswer:'''
        
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def chat(self, query):
        """Main chat function to handle a user query."""
        self.history.append({"role": "user", "content": query})

        rewritten_query = self._rewrite_query(query)
        parent_chunk, child_chunk = self._search(rewritten_query)
        answer = self._generate_answer(rewritten_query, parent_chunk, child_chunk)

        self.history.append({"role": "model", "content": answer})
        return answer

