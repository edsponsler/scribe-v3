import re
import codecs

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
    
    footnote_regex = re.compile(r"(\[\\d+\])\\s(.*?)(?=\\s*\[\\d+\]|$)", re.DOTALL)
    
    footnote_map = {marker: content.strip() for marker, content in footnote_regex.findall(footnote_section)}
    
    print(f"  -> Extracted {len(footnote_map)} footnotes.")
    return main_text, footnote_map

def parse_gutenberg_text(raw_text, sections_file_path):
    """
    Parses a raw Gutenberg text file into its constituent parts using a sections definition file.
    This function treats every section uniformly, prepending a title if it doesn't exist.
    """
    print(f"  -> Parsing text using sections file: {sections_file_path}")
    parsed_content = {}

    # 1. Read the sections file and determine the order
    with open(sections_file_path, 'r') as f:
        lines = f.readlines()
    
    sections_def = {}
    section_order = []
    for line in lines:
        if ':' not in line: continue
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip().strip('"')
        sections_def[key] = value
        base_name = key.split('_')[0]
        if base_name not in section_order:
            section_order.append(base_name)

    # 2. Parse sections sequentially
    current_pos = 0
    for section_name in section_order:
        start_marker = sections_def.get(f"{section_name}_start")
        end_marker = sections_def.get(f"{section_name}_end")

        if not start_marker or not end_marker:
            print(f"  -> Warning: Markers for section '{section_name}' not found. Skipping.")
            continue

        start_marker = codecs.decode(start_marker, 'unicode_escape')
        end_marker = codecs.decode(end_marker, 'unicode_escape')

        start_pos = raw_text.find(start_marker, current_pos)
        
        if start_pos == -1:
            print(f"  -> Warning: Could not find start marker for section '{section_name}'. Skipping.")
            continue

        if end_marker == "End Of File":
            end_pos = len(raw_text)
        else:
            # Start searching for the end marker *after* the start marker
            end_pos = raw_text.find(end_marker, start_pos + len(start_marker))

        if end_pos == -1:
            print(f"  -> Warning: Could not find end marker for section '{section_name}'. Skipping.")
            continue
            
        # Extract content
        content_start = start_pos + len(start_marker)
        section_content = raw_text[content_start:end_pos].strip()
        
        # Prepend title if it's not already there
        first_line = section_content.split('\n', 1)[0].strip()
        if first_line.upper() != section_name.upper():
            section_content = f"{section_name.upper()}\n\n{section_content}"

        parsed_content[section_name] = section_content
        current_pos = end_pos

    return parsed_content

def chunk_text_by_paragraph(text):
    """
    Splits a given text into chunks based on paragraphs, filtering out headings.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    
    good_chunks = []
    for p in paragraphs:
        p_stripped = p.strip()
        if len(p_stripped) > 100 and not p_stripped.isupper():
            good_chunks.append(p_stripped)
            
    return good_chunks


