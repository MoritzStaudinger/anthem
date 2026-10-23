import hashlib
import os
from datetime import datetime
import pymupdf as fitz  # PyMuPDF (`fitz` alias was dropped in pymupdf 1.26+)


def extract_year(date_str):
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.year
    except (ValueError, TypeError):
        return None

def remove_duplicates_by_doi(nodes):
    seen_dois = set()
    unique_nodes = []

    for nws in nodes:
        doi = nws.node.metadata.get('doi')
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            unique_nodes.append(nws)

    return unique_nodes

def remove_duplicates_by_doi_dict(papers):
    seen_dois = set()
    unique_papers = []

    for paper in papers:
        doi = paper.get('DOI')
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            unique_papers.append(paper)

    return unique_papers

def extract_title_abstract_fulltext(pdf_path):
    doc = fitz.open(pdf_path)

    title = ""
    abstract = ""
    fulltext = ""

    # Extract fulltext from all pages
    fulltext = "\n".join(page.get_text() for page in doc)

    # Only analyze first page for title/abstract
    blocks = doc[0].get_text("dict")["blocks"]
    text_blocks = []

    for b in blocks:
        if "lines" in b:
            block_text = ""
            max_fontsize = 0
            for line in b["lines"]:
                for span in line["spans"]:
                    block_text += span["text"] + " "
                    max_fontsize = max(max_fontsize, span["size"])
            block_text = block_text.strip()
            if block_text:
                text_blocks.append((block_text, max_fontsize))

    # Sort by font size descending
    sorted_blocks = sorted(text_blocks, key=lambda x: -x[1])

    # Assume largest text block is title
    if sorted_blocks:
        title = sorted_blocks[0][0]

    # Look for abstract block heuristically: between title and introduction
    lower_blocks = [b[0].lower() for b in text_blocks]
    for i, (block, _) in enumerate(text_blocks):
        lower = block.lower()
        if "introduction" in lower or lower.startswith("1 "):
            intro_index = i
            break
    else:
        intro_index = len(text_blocks)

    abstract_candidates = [
        text_blocks[i][0] for i in range(1, intro_index)
        if len(text_blocks[i][0].split()) > 30
    ]
    abstract = " ".join(abstract_candidates)

    return title.strip(), abstract.strip(), fulltext.strip()


def compute_upload_hash(file_storage):
    """
    Compute a SHA256 hash of the uploaded file's contents.
    `file_storage` is an instance of Werkzeug's FileStorage (from request.files).
    """
    hasher = hashlib.sha256()
    file_storage.stream.seek(0)  # Ensure we're at the beginning of the file

    # Read the file in chunks to avoid loading the whole file into memory
    for chunk in iter(lambda: file_storage.stream.read(8192), b""):
        hasher.update(chunk)

    file_storage.stream.seek(0)  # Reset the stream position so it can be read again later
    return hasher.hexdigest()