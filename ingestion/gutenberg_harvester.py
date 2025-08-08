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
from dotenv import load_dotenv

# --- Vertex AI & BigQuery Imports ---
import vertexai
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from ingestion.text_processor import parse_gutenberg_text, extract_footnotes, chunk_text_by_paragraph, chunk_text_by_chapter

def get_book_metadata(book_id):
    """Fetches book metadata from the Gutendex API."""
    try:
        url = f"http://gutendex.com/books/{book_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata for book {book_id} from Gutendex: {e}")
        return None

def map_gutenberg_to_schema(book_metadata, chunk_text, chunk_index, source_section="main_text"):
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
        "chunk_level": "paragraph",
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

def _process_text_section(section_name, section_text, book_metadata, embedding_model, bq_client, cache_table_id):
    """Helper function to run the small-to-big indexing on a section of text."""
    if not section_text.strip():
        print(f"  -> Section '{section_name}' is empty. Skipping.")
        return None, None, None

    print(f"  -> Processing section: {section_name}")
    # 1. Extract footnotes before chunking
    text_sans_footnotes, footnote_map = extract_footnotes(section_text)

    # 2. Create parent chunks (chapters or logical sections)
    parent_chunks = chunk_text_by_chapter(text_sans_footnotes)
    if not parent_chunks:
        print(f"  -> No parent chunks found for section '{section_name}'. Skipping.")
        return None, None, None

    # 3. Create child chunks (paragraphs)
    child_chunks = []
    for parent_chunk in parent_chunks:
        paragraphs = chunk_text_by_paragraph(parent_chunk)
        child_chunks.extend(paragraphs)
    if not child_chunks:
        print(f"  -> No child chunks found for section '{section_name}'. Skipping.")
        return None, None, None

    # 4. Create Retrieval Store (Parent Chunks + Footnote Map)
    retrieval_store = {}
    for i, parent_text in enumerate(parent_chunks):
        parent_id = str(uuid.uuid4())
        retrieval_store[parent_id] = {
            "id": parent_id,
            "source_id": f"gutenberg:{book_metadata.get('id')}",
            "source_section": section_name,
            "title": book_metadata.get("title"),
            "canonical_reference": f"{book_metadata.get('title')} - {section_name} - Section {i + 1}",
            "text": parent_text,
            "footnote_map": footnote_map
        }

    # 5. Create Search Metadata (for child chunks)
    search_metadata_list = []
    for i, child_text in enumerate(child_chunks):
        # Find the parent ID for this child chunk
        parent_id = None
        for pid, parent_data in retrieval_store.items():
            if child_text in parent_data['text']:
                parent_id = pid
                break
        
        child_metadata = map_gutenberg_to_schema(book_metadata, child_text, i, section_name)
        child_metadata['parent_chunk_id'] = parent_id
        search_metadata_list.append(child_metadata)

    # 6. Get Embeddings for child chunks
    child_texts = [m['text'] for m in search_metadata_list]
    child_embeddings = get_embeddings_with_cache(embedding_model, bq_client, cache_table_id, child_texts)

    return retrieval_store, search_metadata_list, child_embeddings

def process_book(book_id, embedding_model, bq_client, cache_table_id):
    """
    Downloads, parses, chunks, and embeds a book with multiple sections.
    """
    try:
        # 1. Get API Metadata and Raw Text
        print(f"  -> Fetching metadata for book {book_id}...")
        api_metadata = get_book_metadata(book_id)
        if not api_metadata: return {}

        book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"  -> Downloading {book_url}")
        response = requests.get(book_url, timeout=15)
        response.raise_for_status()
        raw_text = response.text

        # 2. Parse the raw text into sections
        parsed_content = parse_gutenberg_text(raw_text)

        # 3. Save Book-Level Metadata
        book_metadata = {
            "api_metadata": api_metadata,
            "gutenberg_header": parsed_content["header"],
            "gutenberg_license": parsed_content["license"]
        }
        metadata_path = f"app/local_book_metadata_{book_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(book_metadata, f, indent=2)
        print(f"  -> Book-level metadata saved to '{metadata_path}'")

        # 4. Process and Save Glossary
        if parsed_content["glossary"]:
            # Simple key-value parsing for glossary
            glossary_text = parsed_content["glossary"].replace("GLOSSARY", "").strip()
            glossary_map = dict(re.findall(r'(.*?)\s+(.*)', glossary_text))
            glossary_path = f"app/local_glossary_{book_id}.pkl"
            with open(glossary_path, 'wb') as f:
                pickle.dump(glossary_map, f)
            print(f"  -> Glossary saved to '{glossary_path}'")

        # 5. Process Main Text and Appendix
        main_text_data = _process_text_section("main_text", parsed_content["main_text"], api_metadata, embedding_model, bq_client, cache_table_id)
        appendix_data = _process_text_section("appendix", parsed_content["appendix"], api_metadata, embedding_model, bq_client, cache_table_id)

        # 6. Perform QC Check with Table of Contents
        if parsed_content["contents"]:
            print("  -> Performing QC check with Table of Contents...")
            toc_items = [line.strip() for line in parsed_content["contents"].split('\n') if line.strip() and line.strip() != "CONTENTS"]
            found_titles = [item['canonical_reference'].split(' - ')[-1] for item in main_text_data[0].values()] if main_text_data[0] else []
            # A simple check to see if TOC items are mentioned in found titles
            # This is a basic check and can be improved
            match_count = sum(1 for item in toc_items if any(item in title for title in found_titles))
            print(f"  -> QC Check: {match_count}/{len(toc_items)} Table of Contents items appear to be referenced in the parsed sections.")

        return {
            "main_text": main_text_data,
            "appendix": appendix_data
        }

    except Exception as e:
        print(f"An error occurred while processing book {book_id}: {e}")
        return {}

def main():
    """Main function to orchestrate the full ingestion process."""
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    cache_table_id = os.getenv("BIGQUERY_CACHE_TABLE")
    
    if not all([project_id, location, cache_table_id]):
        print("Error: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and BIGQUERY_CACHE_TABLE must be set in .env")
        return

    print("Configuration loaded successfully.")
    vertexai.init(project=project_id, location=location)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    bq_client = bigquery.Client(project=project_id)
    print("Vertex AI and BigQuery clients initialized.")

    manifest_path = os.path.join(os.path.dirname(__file__), 'gutenberg_manifest.txt')
    book_ids = load_book_ids_from_manifest(manifest_path)
    if not book_ids: return

    print(f"\nStarting ingestion for {len(book_ids)} books (processing first one only)...")
    for book_id in book_ids[:1]:
        print(f"\n--- Processing book ID: {book_id} ---")
        
        # --- Clean up all potential old files ---
        # ... (add cleanup logic here if needed)

        processed_data = process_book(book_id, embedding_model, bq_client, cache_table_id)

        # --- Save all generated files ---
        for section_name, section_data in processed_data.items():
            retrieval_store, search_metadata, search_embeddings = section_data
            if retrieval_store and search_metadata and search_embeddings:
                # Save Retrieval Store (parent chunks)
                retrieval_path = f"app/local_{section_name}_retrieval_{book_id}.pkl"
                with open(retrieval_path, 'wb') as f:
                    pickle.dump(retrieval_store, f)
                print(f"  -> Retrieval store for '{section_name}' saved to '{retrieval_path}'")

                # Save Search Index (child chunk embeddings)
                db_vectors = np.array(search_embeddings, dtype=np.float32)
                index = faiss.IndexFlatL2(db_vectors.shape[1])
                index.add(db_vectors)
                index_path = f"app/local_{section_name}_search_{book_id}.faiss"
                faiss.write_index(index, index_path)
                print(f"  -> Search index for '{section_name}' saved to '{index_path}'")

                # Save Search Metadata (child chunk metadata)
                search_meta_path = f"app/local_{section_name}_search_{book_id}.pkl"
                with open(search_meta_path, 'wb') as f:
                    pickle.dump(search_metadata, f)
                print(f"  -> Search metadata for '{section_name}' saved to '{search_meta_path}'")

        time.sleep(1) # Be nice to the API

    print("\nGutenberg harvester process complete.")


if __name__ == "__main__":
    main()
