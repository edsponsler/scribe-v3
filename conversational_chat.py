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
        self.glossary = {}
        self.book_metadata = {}
        self._load_index()

    def _load_index(self):
        try:
            print(f"--- Loading and Preparing Index and Metadata for source: {self.source} ---")
            
            if self.source == "genesis":
                self.search_indexes['main'] = faiss.read_index(os.path.join(os.path.dirname(__file__), 'app', 'local_verses_genesis.faiss'))
                with open(os.path.join(os.path.dirname(__file__), 'app', 'local_verses_genesis.pkl'), 'rb') as f:
                    self.search_metadatas['main'] = pickle.load(f)
                with open(os.path.join(os.path.dirname(__file__), 'app', 'local_chapters_genesis.pkl'), 'rb') as f:
                    self.retrieval_stores['main'] = pickle.load(f)
                with open(os.path.join(os.path.dirname(__file__), 'app', 'local_book_metadata_genesis.json'), 'r') as f:
                    self.book_metadata = json.load(f)

            elif self.source == "gutenberg":
                BOOK_ID = "2680"
                app_dir = os.path.join(os.path.dirname(__file__), 'app')
                
                search_files = glob.glob(os.path.join(app_dir, f'local_*_search_{BOOK_ID}.faiss'))
                
                for search_file_path in search_files:
                    filename = os.path.basename(search_file_path)
                    match = re.search(f'local_(.+)_search_{BOOK_ID}\.faiss', filename)
                    if not match:
                        continue
                    section_name = match.group(1)

                    search_metadata_path = os.path.join(app_dir, f'local_{section_name}_search_{BOOK_ID}.pkl')
                    retrieval_store_path = os.path.join(app_dir, f'local_{section_name}_retrieval_{BOOK_ID}.pkl')

                    if os.path.exists(search_metadata_path) and os.path.exists(retrieval_store_path):
                        self.search_indexes[section_name] = faiss.read_index(search_file_path)
                        with open(search_metadata_path, 'rb') as f:
                            self.search_metadatas[section_name] = pickle.load(f)
                        with open(retrieval_store_path, 'rb') as f:
                            self.retrieval_stores[section_name] = pickle.load(f)
                        print(f"  -> Loaded data for section: {section_name}")

                glossary_path = os.path.join(app_dir, f'local_glossary_{BOOK_ID}.pkl')
                if os.path.exists(glossary_path):
                    with open(glossary_path, 'rb') as f:
                        self.glossary = pickle.load(f)
                
                metadata_path = os.path.join(app_dir, f'local_book_metadata_{BOOK_ID}.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.book_metadata = json.load(f)
            else:
                raise ValueError(f"Unknown source: {self.source}")

            print(f"  -> All data for source '{self.source}' loaded successfully.")

        except Exception as e:
            print(f"Error loading local index: {e}")

    def _rewrite_query(self, query):
        """Rewrites a potentially ambiguous query into a self-contained one."""
        if not self.history:
            return query, 0, 0

        history_string = "\n".join([f"{turn['role']}: {turn['content']}" for turn in self.history])
        prompt = f'''Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.\n\nConversation History:\n{history_string}\n\nFollow-up Question: {query}\n\nStandalone Question:'''
        
        response = self.model.generate_content(prompt)
        
        rewritten_query = response.text.strip()
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        print(f"  -> Rewritten Query: {rewritten_query} (Tokens: In={input_tokens}, Out={output_tokens})")
        return rewritten_query, input_tokens, output_tokens

    def _search(self, query):
        """Performs a unified vector search across all available indexes for the source."""
        print(f"  -> Searching for: '{query}'")
        query_embedding = self.embedding_model.get_embeddings([query])[0].values
        k = 1

        if self.source == "genesis":
            search_index = self.search_indexes['main']
            distances, indices = search_index.search(np.array([query_embedding], dtype=np.float32), k)
            if not indices.any():
                return None, None
            
            top_index = indices[0][0]
            most_relevant_chunk_metadata = self.search_metadatas['main'][top_index]
            parent_id = most_relevant_chunk_metadata.get('parent_chunk_id')

            if not parent_id:
                print("  -> Could not find parent ID for the most relevant chunk.")
                return most_relevant_chunk_metadata, most_relevant_chunk_metadata

            retrieved_parent = self.retrieval_stores.get(parent_id)
            if not retrieved_parent:
                if self.retrieval_stores.get('main'):
                    retrieved_parent = self.retrieval_stores['main'].get(parent_id)
                if not retrieved_parent:
                    print(f"  -> Could not find parent chunk with ID {parent_id} in any retrieval store.")
                    return None, None

            retrieved_child = None
            for chunk in retrieved_parent.get('child_chunks', []):
                if chunk.get('chunk_id') == most_relevant_chunk_metadata.get('chunk_id'):
                    retrieved_child = chunk
                    break

            if not retrieved_child:
                print("  -> Could not find the specific child chunk within the parent.")
                retrieved_child = most_relevant_chunk_metadata

            print(f"  -> Retrieved 1 parent chunk and 1 child chunk from Genesis index.")
            return retrieved_parent, retrieved_child

        available_sections = [s for s in self.search_indexes.keys() if s != 'introduction']
        print(f"  -> Searching across sections: {available_sections}")

        top_results = []
        for section_name in available_sections:
            search_index = self.search_indexes[section_name]
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

        best_result = min(top_results, key=lambda x: x['distance'])
        print(f"  -> Top result found in section: '{best_result['section']}' (Distance: {best_result['distance']:.4f})")

        top_section = best_result['section']
        top_index = best_result['index']
        most_relevant_chunk_metadata = self.search_metadatas[top_section][top_index]
        parent_id = most_relevant_chunk_metadata.get('parent_chunk_id')

        if not parent_id:
            print("  -> Could not find parent ID for the most relevant chunk.")
            return None, None

        retrieved_parent = self.retrieval_stores[top_section].get(parent_id)

        retrieved_child = None
        if retrieved_parent:
            for chunk in retrieved_parent.get('child_chunks', []):
                if chunk.get('chunk_id') == most_relevant_chunk_metadata.get('chunk_id'):
                    retrieved_child = chunk
                    break

        if not retrieved_child:
            print("  -> Could not find the specific child chunk within the parent.")
            retrieved_child = most_relevant_chunk_metadata

        print(f"  -> Retrieved 1 parent chunk and 1 child chunk based on unified search.")
        return retrieved_parent, retrieved_child

    def _generate_answer(self, query, parent_chunk, child_chunk):
        """Generates a final answer based on the retrieved context, footnotes, and glossary."""
        if not parent_chunk or not child_chunk:
            return "I could not find a relevant passage to answer your question.", 0, 0

        if self.source == "gutenberg" or self.source == "genesis":
            parent_text = parent_chunk.get('text', '')
            print(f"  -> Using full parent chunk text as context ({len(parent_text)} chars).")
        else:
            parent_text = child_chunk.get('text', '')
            print("  -> Using single chunk context.")

        footnote_map = parent_chunk.get('footnote_map', {})
        
        referenced_footnotes = {}
        for marker, text in footnote_map.items():
            if marker in parent_text:
                referenced_footnotes[marker] = text

        defined_terms = {}
        if self.glossary:
            for term, definition in self.glossary.items():
                if re.search(r'\b' + re.escape(term) + r'\b', parent_text, re.IGNORECASE):
                    defined_terms[term] = definition

        if self.source == 'genesis':
            verses_with_numbers = []
            for i, chunk in enumerate(parent_chunk.get('child_chunks', [])):
                verse_ref = chunk.get('canonical_reference', '')
                verses_with_numbers.append(f"({verse_ref}) {chunk.get('text', '')}")
            context_string = " ".join(verses_with_numbers)
            print(f"  -> Using numbered verses as context.")
            prompt_instruction = "Where appropriate, provide an inline citation in parenthesis to the verse(s) from which your answer is sourced, in the format (Book Chapter:Verse)."
        elif self.source == 'gutenberg':
            logical_section = parent_chunk.get('source_section', 'Unknown Section')
            section_name = logical_section.replace("_", " ").title()
            all_child_chunks = parent_chunk.get('child_chunks', [])
            try:
                child_index = next(i for i, chunk in enumerate(all_child_chunks) if chunk.get('chunk_id') == child_chunk.get('chunk_id'))
            except StopIteration:
                child_index = -1

            paragraphs_with_numbers = []
            if child_index != -1:
                start_index = max(0, child_index - 1)
                end_index = min(len(all_child_chunks), child_index + 2)
                context_window_indices = range(start_index, end_index)
                print(f"  -> Using a sliding window of {len(context_window_indices)} paragraphs for context.")

                for i in context_window_indices:
                    chunk = all_child_chunks[i]
                    para_num = i + 1
                    paragraphs_with_numbers.append(f"({section_name}, paragraph {para_num}) {chunk.get('text', '')}")
            else:
                paragraphs_with_numbers.append(f"({section_name}, paragraph 1) {child_chunk.get('text', '')}")
                print("  -> Warning: Could not find child in parent, using single chunk context.")

            context_string = "\n\n".join(paragraphs_with_numbers)
            prompt_instruction = f"Where appropriate, provide an inline citation in parenthesis to the paragraph number from which your answer is sourced, in the format ({section_name}, paragraph [number])."
        else:
            context_string = f"Text: {parent_text}\n\n"
            prompt_instruction = "Where appropriate, provide an inline citation in parenthesis to the section or paragraph from which your answer is sourced."

        if referenced_footnotes:
            footnote_string = "\n".join([f"{marker}: {text}" for marker, text in referenced_footnotes.items()])
            context_string += f"RELEVANT FOOTNOTES:\n---\n{footnote_string}\n---\n\n"

        if defined_terms:
            glossary_string = "\n".join([f"- {term}: {definition}" for term, definition in defined_terms.items()])
            context_string += f"RELEVANT GLOSSARY TERMS:\n---\n{glossary_string}\n---\n\n"

        prompt = f'''{CORE_PERSONA_PROMPT}\n\nAnswer the following question using only the provided context. Be concise and direct. {prompt_instruction}\n\nContext:\n---\n{context_string}---\n\nQuestion: {query}\n\nAnswer:'''
        
        response = self.model.generate_content(prompt)
        
        answer = response.text.strip()
        
        if self.book_metadata:
            title = self.book_metadata.get('title', 'Unknown Title')
            author = self.book_metadata.get('author', 'N/A')
            publisher = self.book_metadata.get('publisher', 'N/A')
            pub_year = self.book_metadata.get('publication_year', 'N/A')
            final_reference = f"*{title}*, by {author}. {publisher}, {pub_year}."
        else:
            final_reference = child_chunk.get('canonical_reference', 'N/A').split(' ')[0]

        final_answer = f"{answer}\n\nReference: {final_reference}"
        
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        return final_answer, input_tokens, output_tokens

    def chat(self, query):
        """Main chat function to handle a user query."""
        self.history.append({"role": "user", "content": query})

        rewritten_query, rewrite_in, rewrite_out = self._rewrite_query(query)
        
        parent_chunk, child_chunk = self._search(rewritten_query)
        
        answer, answer_in, answer_out = self._generate_answer(rewritten_query, parent_chunk, child_chunk)

        total_input_tokens = rewrite_in + answer_in
        total_output_tokens = rewrite_out + answer_out
        print(f"--- Token Usage for this query ---")
        print(f"  - Rewrite Step: In={rewrite_in}, Out={rewrite_out}")
        print(f"  - Answer Step:  In={answer_in}, Out={answer_out}")
        print(f"  - Total:        In={total_input_tokens}, Out={total_output_tokens}")
        print(f"---------------------------------")

        self.history.append({"role": "model", "content": answer})
        return answer

if __name__ == "__main__":
    # This is a simple command-line interface for testing the agent.
    load_dotenv()
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
    
    if not PROJECT_ID or not LOCATION:
        print("Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION in your .env file.")
    else:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        parser = argparse.ArgumentParser(description="Conversational Agent CLI")
        parser.add_argument("--source", type=str, default="gutenberg", help="The source to use (e.g., 'genesis' or 'gutenberg')")
        args = parser.parse_args()

        agent = ConversationalAgent("gemini-1.5-flash", source=args.source)
        print(f"Welcome to the SCRIBE v3 CLI. Chatting with source: '{args.source}'. Type 'exit' to quit.")
        
        while True:
            query = input("You: ")
            if query.lower() == 'exit':
                break
            response = agent.chat(query)
            print(f"SCRIBE: {response}")