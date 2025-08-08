import re

def extract_footnotes(text):
    """
    Finds a footnote section, extracts the footnotes into a map,
    and returns the text with the footnote section removed.
    """
    footnote_marker = "FOOTNOTES:"
    footnote_pos = text.rfind(footnote_marker)
    
    if footnote_pos == -1:
        return text, {} # No footnotes found

    main_text = text[:footnote_pos].strip()
    footnote_section = text[footnote_pos + len(footnote_marker):].strip()
    
    # Regex to find footnote definitions, e.g., "[1] Some text."
    footnote_regex = re.compile(r"(\[\d+\])\s(.*?)(?=\s*\[\d+\]|$)", re.DOTALL)
    
    footnote_map = {marker: content.strip() for marker, content in footnote_regex.findall(footnote_section)}
    
    print(f"  -> Extracted {len(footnote_map)} footnotes.")
    return main_text, footnote_map

def parse_gutenberg_text(raw_text):
    """
    Parses a raw Gutenberg text file into its constituent parts.
    Returns a dictionary containing all extracted sections.
    """
    parsed_content = {
        "header": "",
        "contents": "",
        "main_text": "",
        "appendix": "",
        "notes": "",
        "glossary": "",
        "license": ""
    }

    # 1. Isolate Header and License (everything outside the markers)
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_pos = raw_text.find(start_marker)
    end_pos = raw_text.find(end_marker)

    if start_pos != -1:
        parsed_content["header"] = raw_text[:start_pos].strip()
        content_start = raw_text.find('\n', start_pos) + 1
    else:
        content_start = 0 # No header found

    if end_pos != -1:
        parsed_content["license"] = raw_text[end_pos + len(end_marker):].strip()
        core_text = raw_text[content_start:end_pos].strip()
    else:
        core_text = raw_text[content_start:].strip()

    # 2. Sequentially find and slice off each major section from the end
    sections_in_order = [
        ("glossary", "GLOSSARY"),
        ("notes", "NOTES"),
        ("appendix", "APPENDIX")
    ]
    
    for key, marker in sections_in_order:
        # Use rfind to find the last occurrence, which should be the section start
        marker_pos = core_text.rfind(marker)
        # Check if the marker is at the beginning of a line
        if marker_pos > 0 and core_text[marker_pos-1] in ('\n', '\r'):
            section_text = core_text[marker_pos:].strip()
            parsed_content[key] = section_text
            core_text = core_text[:marker_pos].strip()
            print(f"  -> Found and separated the {key.upper()} section.")

    # 3. Find the Table of Contents
    contents_marker = "CONTENTS"
    contents_pos = core_text.find(contents_marker)
    if contents_pos != -1 and core_text[contents_pos-1] in ('\n', '\r'):
        # Find the end of the contents section (usually before the first chapter)
        prose_start_markers = [' INTRODUCTION', 'CHAPTER I', 'BOOK I']
        end_contents_pos = -1
        for marker in prose_start_markers:
            pos = core_text.find(marker, contents_pos)
            if pos != -1:
                if end_contents_pos == -1 or pos < end_contents_pos:
                    end_contents_pos = pos
        
        if end_contents_pos != -1:
            parsed_content["contents"] = core_text[contents_pos:end_contents_pos].strip()
            core_text = core_text[end_contents_pos:].strip()
            print("  -> Found and separated the CONTENTS section.")

    # 4. The remainder is the main text
    parsed_content["main_text"] = core_text.strip()

    return parsed_content

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

def chunk_text_by_chapter(text):
    """
    Splits a given text into chunks based on chapters.
    """
    # This regex looks for "CHAPTER" followed by a space and a Roman numeral or number.
    # It handles variations in spacing and newlines.
    chapters = re.split(r'(CHAPTER\s+[IVXLC\d]+\.?)', text, flags=re.IGNORECASE)
    
    # The result of the split will be [prologue, chapter_marker_1, chapter_content_1, chapter_marker_2, ...]
    # We need to recombine the marker with its content.
    
    chunked_chapters = []
    # The first element is the text before the first chapter marker (prologue, etc.)
    # We can choose to ignore it or handle it as a separate chunk if it's substantial.
    if chapters[0].strip():
        chunked_chapters.append(chapters[0].strip())
        
    # Iterate through the rest of the list, pairing markers with content
    for i in range(1, len(chapters), 2):
        if i + 1 < len(chapters):
            chapter_title = chapters[i].strip()
            chapter_content = chapters[i+1].strip()
            # We prepend the title to the content to form a complete chapter chunk
            full_chapter = f"{chapter_title}\n\n{chapter_content}"
            chunked_chapters.append(full_chapter)
            
    return chunked_chapters
