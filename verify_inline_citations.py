import re
import sys
sys.path.append('ingestion')
from ingestion.text_processor import chunk_text_by_chapter, parse_gutenberg_text

def extract_paragraphs_from_book(file_path, book_title, paragraphs_to_extract):
    """Extracts specific numbered paragraphs from a given book in the text."""
    print(f"--- Extracting paragraphs {paragraphs_to_extract} from '{book_title}' ---")
    try:
        with open(file_path, 'r') as f:
            raw_text = f.read()

        parsed_content = parse_gutenberg_text(raw_text)
        main_text = parsed_content.get("main_text", "")
        parent_chunks = chunk_text_by_chapter(main_text)

        target_book_text = ""
        for chunk in parent_chunks:
            if chunk.strip().startswith(book_title):
                target_book_text = chunk
                break
        
        if not target_book_text:
            print(f"Error: Could not find book titled '{book_title}'.")
            return

        # Split the book into paragraphs by Roman numerals
        # This regex looks for a Roman numeral at the beginning of a line
        paragraph_chunks = re.split(r'\n([IVXLCDM]+)\.\s', target_book_text)

        print(f"--- Content of Specified Paragraphs ---")
        # The list is structured as [intro, numeral_1, content_1, numeral_2, content_2, ...]
        for i in range(1, len(paragraph_chunks), 2):
            numeral = paragraph_chunks[i]
            content = paragraph_chunks[i+1].strip()
            if numeral in paragraphs_to_extract:
                print(f"\n--- PARAGRAPH {numeral} ---")
                print(content)
        print("\n-----------------------------------------")

    except FileNotFoundError:
        print(f"Error: Test file not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_file = "ingestion/raw/pg2680.txt"
    book_to_inspect = "THE FOURTH BOOK"
    paragraphs = ['II', 'III', 'IV']
    extract_paragraphs_from_book(test_file, book_to_inspect, paragraphs)
