import os
import time
import requests
import json
import uuid
import hashlib
import pickle
import numpy as np
import faiss
import re
import tempfile
import shutil
from dotenv import load_dotenv

# --- Vertex AI & BigQuery Imports ---
import vertexai
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from ingestion.text_processor import parse_gutenberg_text, extract_footnotes, chunk_text_by_paragraph

# --- Global Generative Model ---
# Initialized in main()
generative_model = None

def get_book_metadata(header_text):
    """Extracts book metadata from the header text."""
    title_match = re.search(r"Title: (.+)", header_text)
    author_match = re.search(r"Author: (.+)", header_text)
    release_date_match = re.search(r"Release date: (.+)", header_text)

    return {
        "title": title_match.group(1).strip() if title_match else "Unknown Title",
        "author": author_match.group(1).strip() if author_match else "N/A",
        "publication_year": release_date_match.group(1).split('[')[0].strip() if release_date_match else "N/A"
    }

def generate_summary(text_to_summarize, prompt="Summarize the following text in 1-2 sentences, capturing the main themes and arguments:"):
    """Generates a brief summary of a text using a generative model."""
    global generative_model
    if not generative_model:
        print("Error: Generative model not initialized.")
        return ""
    
    full_prompt = f"{prompt}\n\n---\n{text_to_summarize}\n---"
    try:
        response = generative_model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return ""

def get_summary_with_cache(bq_client, cache_table_id, texts):
    """Gets summaries for a list of text chunks, using a BigQuery cache."""
    if not texts:
        return []
    print(f"  -> Getting summaries for {len(texts)} texts, using BigQuery cache...")
    summaries = []
    api_calls = 0

    hashes_to_check = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    
    query = f"SELECT chunk_hash, summary FROM `{cache_table_id}` WHERE chunk_hash IN UNNEST(@hashes)"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("hashes", "STRING", hashes_to_check)]
    )
    cached_results = {row.chunk_hash: row.summary for row in bq_client.query(query, job_config=job_config)}
    
    new_texts_for_batch = []
    hash_to_text_map = {}
    for text, text_hash in zip(texts, hashes_to_check):
        if text_hash not in cached_results:
            new_texts_for_batch.append(text)
            hash_to_text_map[text_hash] = text
    
    if new_texts_for_batch:
        print(f"  -> Generating {len(new_texts_for_batch)} new summaries...")
        rows_to_insert = []
        for text_hash, text_to_summarize in hash_to_text_map.items():
            api_calls += 1
            summary = generate_summary(text_to_summarize)
            cached_results[text_hash] = summary
            rows_to_insert.append({"chunk_hash": text_hash, "summary": summary})
        
        if rows_to_insert:
            errors = bq_client.insert_rows_json(cache_table_id, rows_to_insert)
            if errors: print(f"Error inserting into summary cache: {errors}")

    for text_hash in hashes_to_check:
        summaries.append(cached_results[text_hash])

    print(f"  -> Summaries retrieved. Made {api_calls} new API calls.")
    return summaries

def map_gutenberg_to_schema(book_metadata, chunk_text, chunk_index, source_section="main_text", chunk_level="paragraph"):
    """Maps Gutenberg data to the SCRIBE metadata schema."""
    book_id = book_metadata.get("id")
    authors = book_metadata.get("authors", [])
    author_names = [author["name"] for author in authors]
    publication_date = authors[0].get("death_year") if authors else None

    mapped_data = {
        "id": str(uuid.uuid4()),
        "source_id": f"gutenberg:{book_id}",
        "source_section": source_section,
        "title": book_metadata.get("title"),
        "author": author_names,
        "publisher": "Project Gutenberg",
        "publication_date": publication_date,
        "source_url": f"https://www.gutenberg.org/ebooks/{book_id}",
        "language": book_metadata.get("languages", ["en"])[0],
        "canonical_reference": f"{book_metadata.get('title')} - {source_section} - Chunk {chunk_index + 1}",
        "text_type": "book",
        "chunk_level": chunk_level,
        "text": chunk_text
    }
    return {k: v for k, v in mapped_data.items() if v is not None}

def load_book_ids_from_manifest(manifest_path):
    """Loads book IDs from a text manifest file."""
    try:
        with open(manifest_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        return []

def get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, chunks):
    """Gets embeddings for a list of text chunks, using a BigQuery cache."""
    if not chunks:
        return []
    print(f"  -> Getting embeddings for {len(chunks)} chunks, using BigQuery cache...")
    embedding_vectors = []
    api_calls = 0
    BATCH_SIZE = 50

    for i in range(0, len(chunks), BATCH_SIZE):
        chunk_batch = chunks[i:i + BATCH_SIZE]
        hashes_to_check = [hashlib.sha256(text.encode()).hexdigest() for text in chunk_batch]
        
        query = f"SELECT chunk_hash, embedding FROM `{cache_table_id}` WHERE chunk_hash IN UNNEST(@hashes)"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("hashes", "STRING", hashes_to_check)]
        )
        cached_results = {row.chunk_hash: row.embedding for row in bq_client.query(query, job_config=job_config)}
        
        new_texts_for_batch = []
        hash_to_text_map = {}
        for text, text_hash in zip(chunk_batch, hashes_to_check):
            if text_hash not in cached_results:
                new_texts_for_batch.append(text)
                hash_to_text_map[text_hash] = text
        
        if new_texts_for_batch:
            api_calls += len(new_texts_for_batch)
            inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in new_texts_for_batch]
            embedding_responses = embedding_model.get_embeddings(inputs)
            
            rows_to_insert = []
            for text_hash, embedding in zip(hash_to_text_map.keys(), embedding_responses):
                embedding_vector = embedding.values
                cached_results[text_hash] = embedding_vector
                rows_to_insert.append({"chunk_hash": text_hash, "embedding": embedding_vector})
            
            if rows_to_insert:
                errors = bq_client.insert_rows_json(cache_table_id, rows_to_insert)
                if errors: print(f"Error inserting into cache: {errors}")

        for text_hash in hashes_to_check:
            embedding_vectors.append(cached_results[text_hash])

    print(f"  -> Embeddings retrieved. Made {api_calls} new API calls.")
    return embedding_vectors

def _process_text_section(section_name, section_text, book_metadata, embedding_model, bq_client, embedding_cache_table_id, summary_cache_table_id):
    """Helper function to run the small-to-big indexing on a section of text."""
    if not section_text.strip():
        print(f"  -> Section '{section_name}' is empty. Skipping.")
        return None, None, None, None, None

    print(f"  -> Processing section: {section_name}")
    text_sans_footnotes, footnote_map = extract_footnotes(section_text)
    parent_chunks = [text_sans_footnotes]

    child_chunks = []
    for parent_chunk in parent_chunks:
        paragraphs = chunk_text_by_paragraph(parent_chunk)
        child_chunks.extend(paragraphs)
    if not child_chunks:
        print(f"  -> No child chunks found for section '{section_name}'. Skipping.")
        return None, None, None, None, None

    summaries = get_summary_with_cache(bq_client, summary_cache_table_id, parent_chunks)
    summary_text = summaries[0] if summaries else ""
    
    retrieval_store = {}
    summary_metadata_list = []
    search_metadata_list = []

    for i, parent_text in enumerate(parent_chunks):
        parent_id = str(uuid.uuid4())
        first_line = parent_text.split('\n', 1)[0].strip()
        section_title = first_line if len(first_line) < 100 and first_line.isupper() else section_name.replace("_", " ").title()

        retrieval_store[parent_id] = {
            "id": parent_id,
            "source_id": f"gutenberg:{book_metadata.get('id')}",
            "source_section": section_name,
            "title": book_metadata.get("title"),
            "canonical_reference": f"{book_metadata.get('title')}, {section_title}",
            "text": parent_text,
            "summary": summary_text,
            "footnote_map": footnote_map,
            "child_chunks": []
        }
        
        summary_meta = map_gutenberg_to_schema(book_metadata, summary_text, i, section_name, chunk_level="summary")
        summary_meta['parent_chunk_ids'] = [parent_id]
        summary_metadata_list.append(summary_meta)

        paragraphs = chunk_text_by_paragraph(parent_text)
        for i, child_text in enumerate(paragraphs):
            child_metadata = map_gutenberg_to_schema(book_metadata, child_text, i, section_name)
            child_metadata['parent_chunk_ids'] = [parent_id]
            child_metadata['chunk_id'] = str(uuid.uuid4())
            search_metadata_list.append(child_metadata)
            retrieval_store[parent_id]['child_chunks'].append(child_metadata)

    child_texts = [m['text'] for m in search_metadata_list]
    child_embeddings = get_embeddings_with_cache(embedding_model, bq_client, embedding_cache_table_id, child_texts)
    
    summary_texts = [m['text'] for m in summary_metadata_list]
    summary_embeddings = get_embeddings_with_cache(embedding_model, bq_client, embedding_cache_table_id, summary_texts)

    return retrieval_store, search_metadata_list, child_embeddings, summary_metadata_list, summary_embeddings

def process_book(book_id, embedding_model, bq_client, embedding_cache_table_id, summary_cache_table_id, output_dir):
    """
    Downloads, parses, chunks, and embeds a book with multiple sections.
    Writes all output to a specified directory.
    """
    book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(f"  -> Downloading {book_url}")
    response = requests.get(book_url, timeout=15)
    response.raise_for_status()
    raw_text = response.text

    sections_file_path = os.path.join(os.path.dirname(__file__), 'raw', f'pg{book_id}-sections.txt')
    parsed_content = parse_gutenberg_text(raw_text, sections_file_path)

    book_metadata = get_book_metadata(parsed_content.get("HEADER", ""))
    book_metadata["id"] = book_id
    book_metadata["gutenberg_license"] = parsed_content.get("LICENSE", "")
    metadata_path = os.path.join(output_dir, f"local_book_metadata_{book_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(book_metadata, f, indent=2)
    print(f"  -> Book-level metadata saved to '{metadata_path}'")

    if "GLOSSARY" in parsed_content:
        glossary_text = parsed_content["GLOSSARY"].replace("GLOSSARY", "").strip()
        glossary_map = dict(re.findall(r'(.*?)\s+(.*)', glossary_text))
        glossary_path = os.path.join(output_dir, f"local_glossary_{book_id}.pkl")
        with open(glossary_path, 'wb') as f:
            pickle.dump(glossary_map, f)
        print(f"  -> Glossary saved to '{glossary_path}'")

    ignore_list = ["HEADER", "CONTENTS", "NOTES", "GLOSSARY", "LICENSE"]

    for section_name, section_text in parsed_content.items():
        if section_name in ignore_list:
            continue
        
        logical_name = section_name.lower().replace(" ", "_")
        
        print(f"--- Processing new logical section: {logical_name} ---")
        section_data = _process_text_section(
            logical_name, 
            section_text, 
            book_metadata, 
            embedding_model, 
            bq_client, 
            embedding_cache_table_id, 
            summary_cache_table_id
        )
        
        if not section_data: continue
        retrieval_store, search_metadata, search_embeddings, summary_metadata, summary_embeddings = section_data
        
        # Save paragraph-level data
        if retrieval_store and search_metadata and search_embeddings:
            retrieval_path = os.path.join(output_dir, f"local_{logical_name}_retrieval_{book_id}.pkl")
            with open(retrieval_path, 'wb') as f:
                pickle.dump(retrieval_store, f)

            db_vectors = np.array(search_embeddings, dtype=np.float32)
            index = faiss.IndexFlatL2(db_vectors.shape[1])
            index.add(db_vectors)
            index_path = os.path.join(output_dir, f"local_{logical_name}_search_{book_id}.faiss")
            faiss.write_index(index, index_path)

            search_meta_path = os.path.join(output_dir, f"local_{logical_name}_search_{book_id}.pkl")
            with open(search_meta_path, 'wb') as f:
                pickle.dump(search_metadata, f)

        # Save summary-level data
        if summary_metadata and summary_embeddings:
            db_vectors = np.array(summary_embeddings, dtype=np.float32)
            index = faiss.IndexFlatL2(db_vectors.shape[1])
            index.add(db_vectors)
            index_path = os.path.join(output_dir, f"local_{logical_name}_summaries_{book_id}.faiss")
            faiss.write_index(index, index_path)

            summary_meta_path = os.path.join(output_dir, f"local_{logical_name}_summaries_{book_id}.pkl")
            with open(summary_meta_path, 'wb') as f:
                pickle.dump(summary_metadata, f)

def main():
    """Main function to orchestrate the full ingestion process."""
    global generative_model
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    embedding_cache_table_id = os.getenv("BIGQUERY_CACHE_TABLE")
    summary_cache_table_id = os.getenv("BIGQUERY_SUMMARY_CACHE_TABLE")
    generative_model_name = os.getenv("VERTEX_GENERATIVE_MODEL", "gemini-2.5-flash")
    
    if not all([project_id, location, embedding_cache_table_id, summary_cache_table_id]):
        print("Error: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, BIGQUERY_CACHE_TABLE, and BIGQUERY_SUMMARY_CACHE_TABLE must be set in .env")
        return

    print("Configuration loaded successfully.")
    vertexai.init(project=project_id, location=location)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    generative_model = GenerativeModel(generative_model_name)
    bq_client = bigquery.Client(project=project_id)
    print("Vertex AI and BigQuery clients initialized.")

    manifest_path = os.path.join(os.path.dirname(__file__), 'gutenberg_manifest.txt')
    book_ids = load_book_ids_from_manifest(manifest_path)
    if not book_ids: return

    final_output_dir = os.path.join(os.path.dirname(__file__), '..', 'app')

    print(f"\nStarting ingestion for {len(book_ids)} books (processing first one only)...")
    for book_id in book_ids[:1]:
        print(f"\n--- Processing book ID: {book_id} ---")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                process_book(
                    book_id, 
                    embedding_model, 
                    bq_client, 
                    embedding_cache_table_id, 
                    summary_cache_table_id, 
                    temp_dir
                )
                
                # Move files from temp_dir to final_output_dir
                print(f"  -> Moving generated files to {final_output_dir}")
                for item in os.listdir(temp_dir):
                    s = os.path.join(temp_dir, item)
                    d = os.path.join(final_output_dir, item)
                    shutil.move(s, d)
                print("  -> File move complete.")

            except Exception as e:
                print(f"FATAL: An error occurred while processing book {book_id}: {e}")
                # The temp_dir will be automatically cleaned up
                continue # Move to the next book

        time.sleep(1)

    print("\nGutenberg harvester process complete.")


if __name__ == "__main__":
    main()