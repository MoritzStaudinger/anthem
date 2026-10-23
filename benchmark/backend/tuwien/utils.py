import logging
from collections import Counter
from typing import Dict, List, Optional

from backend.tuwien.client import TUWienClient

logger = logging.getLogger(__name__)

def search_tuwien_papers(
    query: str,
    max_results: int = 100,
    from_year: int = None,
    org_units: List[int] = None
) -> List[Dict]:
    """
    Search TU Wien Repositum for papers using organizational unit queries.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        from_year: Starting year for publications
        org_units: List of organizational unit IDs (default: [1601])
    
    Returns:
        List of structured paper data
    """
    client = TUWienClient()
    
    logger.info("searching TU Wien Repositum for: %s", query)

    raw_pubs = client.search_publications_by_keyword(
        keyword=query,
        org_units=org_units,
        from_year=from_year,
        max_results=max_results
    )

    if not raw_pubs:
        logger.info("no publications found for %r", query)
        return []

    logger.info("found %d matching publications", len(raw_pubs))

    structured_pubs = []
    for pub in raw_pubs:
        try:
            parsed = client.parse_metadata(pub)
            if parsed.get("title") and parsed.get("abstract"):
                if len(parsed["abstract"].strip()) > 50:
                    structured_pubs.append(parsed)
        except Exception:
            logger.exception("failed to parse publication")
            continue

    logger.info("structured %d publications with complete metadata", len(structured_pubs))
    return structured_pubs

def get_tuwien_profile(
    topic: str,
    queries: List[str] = None,
    year_start: int = None,
    year_end: int = None,
    org_units: List[int] = None,
    include_researchers: bool = False
) -> Dict:
    """
    Get TU Wien research profile for a topic.
    
    Args:
        topic: Main research topic
        queries: List of specific query strings
        year_start: Filter by start year
        year_end: Filter by end year
        org_units: List of organizational unit IDs
        include_researchers: Include researcher information (not implemented yet)
    
    Returns:
        Dictionary with TU Wien research profile
    """
    client = TUWienClient()
    all_papers = []
    
    # Search by main topic
    logger.info("searching TU Wien for topic: %s", topic)
    papers = search_tuwien_papers(
        topic,
        max_results=50,
        from_year=year_start,
        org_units=org_units
    )
    all_papers.extend(papers)
    
    # Search by queries
    if queries:
        for query in queries:
            logger.debug("searching TU Wien for query: %s", query)
            papers = search_tuwien_papers(
                query,
                max_results=30,
                from_year=year_start,
                org_units=org_units
            )
            all_papers.extend(papers)
    
    # Filter by year_end if specified
    if year_end:
        all_papers = [
            p for p in all_papers
            if not p.get("publication_year") or p["publication_year"] <= year_end
        ]
    
    # Remove duplicates by DOI or title
    seen = set()
    unique_papers = []
    for paper in all_papers:
        # Use DOI if available, otherwise use title
        doi = paper.get("DOI")
        title = paper.get("title")
        key = doi if doi else title
        
        if key and key not in seen:
            seen.add(key)
            unique_papers.append(paper)
        elif not key:
            # If neither DOI nor title, still add it (edge case)
            unique_papers.append(paper)
    
    logger.info("total unique TU Wien papers: %d", len(unique_papers))
    
    # Count publications and top authors
    all_authors = []
    for paper in unique_papers:
        all_authors.extend(paper.get("authors", []))
    
    top_authors = Counter(all_authors).most_common(10)
    top_authors_str = "\n".join(
        f"{name} ({count} publications)" 
        for name, count in top_authors
    ) if top_authors else "N/A"
    
    # Group by year
    year_counts = {}
    for paper in unique_papers:
        year = paper.get("publication_year")
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1
    
    # Get top subjects/keywords
    all_subjects = []
    for paper in unique_papers:
        all_subjects.extend(paper.get("subjects", []))
    top_subjects = Counter(all_subjects).most_common(10)
    
    # Get organizational unit distribution
    all_orgunits = []
    for paper in unique_papers:
        all_orgunits.extend(paper.get("orgunits", []))
    top_orgunits = Counter(all_orgunits).most_common(5)
    
    return {
        "Institution": "TU Wien",
        "Publications": len(unique_papers),
        "PublicationsPerYear": dict(sorted(year_counts.items())),
        "Year Start": year_start,
        "Year End": year_end,
        "Researchers": top_authors_str,
        "TopSubjects": [subj for subj, _ in top_subjects],
        "TopOrganizationalUnits": dict(top_orgunits),
        "papers": unique_papers
    }

def get_all_tuwien_publications(
    from_year: int = None,
    org_units: List[int] = None,
    max_items: int = None
) -> List[Dict]:
    """
    Get all TU Wien publications from specified organizational units.
    Useful for building a comprehensive index.
    
    Args:
        from_year: Starting year
        org_units: List of org unit IDs (default: [1601])
        max_items: Maximum items to retrieve
    
    Returns:
        List of all publications
    """
    client = TUWienClient()
    
    if org_units is None:
        org_units = [1601] 
    
    all_pubs = []
    for org_id in org_units:
        logger.info("fetching publications from org unit %s", org_id)
        raw_pubs = client.query_orgunit_publications(
            org_id,
            from_year=from_year,
            max_items=max_items
        )
        
        for pub in raw_pubs:
            try:
                parsed = client.parse_metadata(pub)
                if parsed.get("title") and parsed.get("abstract"):
                    all_pubs.append(parsed)
            except Exception as e:
                continue
    
    # Remove duplicates
    seen = set()
    unique_pubs = []
    for pub in all_pubs:
        key = pub.get("DOI") or pub.get("title")
        if key and key not in seen:
            seen.add(key)
            unique_pubs.append(pub)
    
    return unique_pubs

def get_research_areas() -> Optional[Dict]:
    """Get research areas from TISS PDB API."""
    client = TUWienClient()
    return client.get_research_areas()

def get_research_units() -> List[Dict]:
    """Get all research units from TISS FPF API."""
    client = TUWienClient()
    return client.get_all_research_units()