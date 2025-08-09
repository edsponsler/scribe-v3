
import os
import requests
import re
from dotenv import load_dotenv
import sys

# Add the 'ingestion' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ingestion'))
from text_processor import parse_gutenberg_text, extract_footnotes, chunk_text_by_paragraph, chunk_text_by_chapter

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

def load_book_ids_from_manifest(manifest_path):
    """Loads book IDs from a text manifest file."""
    try:
        with open(manifest_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        return []

def _process_text_section_dry_run(section_name, section_text, book_metadata):
    """Helper function to run the small-to-big indexing on a section of text."""
    if not section_text.strip():
        print(f"  -> Section '{section_name}' is empty. Skipping.")
        return

    print(f"  -> Processing section: {section_name}")
    text_sans_footnotes, _ = extract_footnotes(section_text)
    parent_chunks = chunk_text_by_chapter(text_sans_footnotes)
    if not parent_chunks:
        print(f"  -> No parent chunks found for section '{section_name}'. Skipping.")
        return

    print(f"    -> Found {len(parent_chunks)} parent chunks (potential books/chapters).")
    for i, parent_text in enumerate(parent_chunks):
        first_line = parent_text.split('\n', 1)[0].strip()
        
        # Corrected heuristic to identify book titles
        book_title_match = re.match(r"THE (FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH) BOOK", first_line, re.IGNORECASE)
        
        if book_title_match:
            section_title = book_title_match.group(0)
        elif len(first_line) < 100 and first_line.isupper():
            section_title = first_line
        else:
            section_title = f"Section {i + 1}"
            
        canonical_reference = f"{book_metadata.get('title')}, {section_title}"
        print(f"      - Identified canonical reference: {canonical_reference}")


def process_book_dry_run(book_id):
    """
    Downloads, parses, and chunks a book with multiple sections, then prints the identified sections.
    """
    try:
        # Use mock metadata to avoid network calls
        api_metadata = {"title": "The Meditations"}

        # Use local file to avoid network calls
        local_path = f"ingestion/raw/pg{book_id}.txt"
        print(f"  -> Reading local file: {local_path}")
        with open(local_path, 'r') as f:
            raw_text = f.read()

        parsed_content = parse_gutenberg_text(raw_text)

        print("\n--- Identified Sections (Dry Run) ---")
        _process_text_section_dry_run("introduction", parsed_content["introduction"], api_metadata)
        _process_text_section_dry_run("main_text", parsed_content["main_text"], api_metadata)
        _process_text_section_dry_run("appendix", parsed_content["appendix"], api_metadata)
        print("--- End of Dry Run ---")

    except Exception as e:
        print(f"An error occurred while processing book {book_id}: {e}")

def main_dry_run():
    """Main function to orchestrate the dry run ingestion process."""
    load_dotenv()
    manifest_path = os.path.join(os.path.dirname(__file__), 'ingestion/gutenberg_manifest.txt')
    book_ids = load_book_ids_from_manifest(manifest_path)
    if not book_ids: return

    print(f"\nStarting dry run for {len(book_ids)} books (processing first one only)...")
    for book_id in book_ids[:1]:
        print(f"\n--- Processing book ID: {book_id} ---")
        process_book_dry_run(book_id)

if __name__ == "__main__":
    main_dry_run()
