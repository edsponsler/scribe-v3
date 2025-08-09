import re
import sys
sys.path.append('ingestion')
from ingestion.text_processor import chunk_text_by_chapter, chunk_text_by_paragraph, parse_gutenberg_text

def test_parent_chunking(file_path):
    """Reads a raw text file, chunks it by book, and prints the first line of each chunk."""
    print(f"--- Testing Parent Chunking Logic on: {file_path} ---")
    try:
        with open(file_path, 'r') as f:
            raw_text = f.read()
        
        parsed_content = parse_gutenberg_text(raw_text)
        main_text = parsed_content.get("main_text", "")

        if not main_text:
            print("Error: Could not find main text to chunk.")
            return

        chunks = chunk_text_by_chapter(main_text)
        
        print(f"-> Found {len(chunks)} parent chunks.")
        print("--- First line of each parent chunk ---")
        for i, chunk in enumerate(chunks):
            first_line = chunk.split('\n', 1)[0].strip()
            print(f"  Chunk {i+1}: {first_line}")
        print("-------------------------------------")

    except FileNotFoundError:
        print(f"Error: Test file not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def test_child_chunking(file_path, book_title_to_inspect):
    """Inspects the paragraph-level child chunks for a specific book."""
    print(f"\n--- Testing Child Chunking Logic for: '{book_title_to_inspect}' ---")
    try:
        with open(file_path, 'r') as f:
            raw_text = f.read()

        parsed_content = parse_gutenberg_text(raw_text)
        main_text = parsed_content.get("main_text", "")
        parent_chunks = chunk_text_by_chapter(main_text)

        target_book_text = ""
        for chunk in parent_chunks:
            if chunk.strip().startswith(book_title_to_inspect):
                target_book_text = chunk
                break
        
        if not target_book_text:
            print(f"Error: Could not find book titled '{book_title_to_inspect}'.")
            return

        child_chunks = chunk_text_by_paragraph(target_book_text)

        print(f"-> Found {len(child_chunks)} paragraph chunks in '{book_title_to_inspect}'.")
        print("--- First 50 characters of the first 5 paragraphs ---")
        for i, chunk in enumerate(child_chunks[:5]):
            print(f"  Paragraph {i+1}: {chunk[:50]}...")
        print("-----------------------------------------------------")

    except FileNotFoundError:
        print(f"Error: Test file not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    test_file = "ingestion/raw/pg2680.txt"
    test_parent_chunking(test_file)
    test_child_chunking(test_file, "THE EIGHTH BOOK")