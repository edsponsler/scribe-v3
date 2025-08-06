import re

def clean_gutenberg_text(text):
    """
    Cleans raw text from Project Gutenberg by removing the header, footer,
    and front matter like the table of contents.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_pos = text.find(start_marker)
    end_pos = text.find(end_marker)

    if start_pos == -1 or end_pos == -1:
        print("  -> Warning: Gutenberg header/footer markers not found. Text may be unclean.")
        return text

    content_start = text.find('\n', start_pos) + 1
    core_text = text[content_start:end_pos].strip()

    prose_markers = [
        'INTRODUCTION', 'FIRST BOOK', 'BOOK I', 
        'CHAPTER 1', 'CHAPTER I', 'PREFACE'
    ]
    
    best_pos = -1
    for marker in prose_markers:
        try:
            match = re.search(r'^\s*' + marker + r'\s*$', core_text, re.MULTILINE)
            if match:
                pos = match.start()
                if best_pos == -1 or pos < best_pos:
                    best_pos = pos
        except re.error:
            continue

    if best_pos != -1:
        print("  -> Found specific prose marker. Skipping table of contents.")
        core_text = core_text[best_pos:]

    return core_text.strip()


def chunk_text_by_paragraph(text):
    """
    Splits a given text into chunks based on paragraphs, filtering out headings.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    
    good_chunks = []
    for p in paragraphs:
        p_stripped = p.strip()
        # Filter out chunks that are too short or are likely headings
        if len(p_stripped) > 100 and not p_stripped.isupper():
            good_chunks.append(p_stripped)
            
    return good_chunks