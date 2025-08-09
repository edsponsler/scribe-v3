import requests
import os
from ingestion.text_processor import parse_gutenberg_text

def test_book_parsing(book_id):
    """
    Performs a dry run of the parsing logic for a given book ID.
    """
    print(f"--- Starting parsing dry run for book ID: {book_id} ---")

    # 1. Fetch raw text from Project Gutenberg
    book_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(f"-> Downloading text from {book_url}")
    try:
        response = requests.get(book_url, timeout=15)
        response.raise_for_status()
        raw_text = response.text
        print("-> Download successful.")
    except requests.RequestException as e:
        print(f"Error: Failed to download book text. {e}")
        return

    # 2. Define path to the sections file
    sections_file_path = os.path.join('ingestion', 'raw', f'pg{book_id}-sections.txt')
    if not os.path.exists(sections_file_path):
        print(f"Error: Sections file not found at '{sections_file_path}'")
        return

    # 3. Call the parsing function from the existing text_processor module
    print(f"-> Parsing text using definitions from '{sections_file_path}'...")
    parsed_content = parse_gutenberg_text(raw_text, sections_file_path)

    # 4. Print a summary of the parsed content for verification
    print("\n--- Parsing Dry Run Results ---")
    for section, content in parsed_content.items():
        if isinstance(content, list):
            print(f"- Section '{section}': Found {len(content)} items.")
            # Just show the first 100 chars of the first item for brevity
            if content:
                print(f"  - First item start: '{content[0][:100]}'...")
        else:
            print(f"- Section '{section}': Found {len(content)} characters.")
            print(f"  - Section start: '{content[:100]}'...")
    print("--- Dry Run Complete ---")

if __name__ == "__main__":
    # Testing the changes made for book 2680
    test_book_parsing("2680")
