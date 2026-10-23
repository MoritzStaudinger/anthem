from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import List, Optional, Tuple

from backend.core.core_api import search_works
from backend.utils.utils import extract_year

class CoreReader(BaseReader):
    def __init__(self) -> None:
        super().__init__()

    def load_data(self, search_query: str, max_results: Optional[int] = 10) -> Tuple[List[Document], List[Document]]:
        search_results = search_works(search_query, limit=max_results)
        full_documents, abstract_documents = [], []

        for paper in search_results:
            doi = paper.get('doi')
            title = paper.get('title', 'No Title')
            abstract = paper.get('abstract', '')
            fulltext = paper.get('fulltext', '').strip()

            metadata = {
                "title": title,
                "authors": paper.get('authors', []),
                "doi": doi,
                "publication_year": extract_year(paper.get('published_date')),
                "update_year": extract_year(paper.get('updated_date')),
                "url": paper.get('url'),
            }

            if doi and fulltext:
                full_documents.append(Document(text=fulltext, metadata=metadata))
            if abstract:
                summary = f"The following is a summary of the paper: {title}\n\nSummary: {abstract}"
                abstract_documents.append(Document(text=summary, metadata=metadata))

        return full_documents #, abstract_documents