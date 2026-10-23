import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

class TUWienClient:
    """
    Client for fetching data from TU Wien's Repositum using organizational unit endpoints.
    Based on working scraper code that uses the orgunit REST API.
    """
    
    BASE_URL = "https://repositum.tuwien.at/rest"
    BASE_URL_FPF = "https://tiss.tuwien.ac.at/api/fpf/rest"
    BASE_URL_PDB = "https://tiss.tuwien.ac.at/api/pdb/rest"
    
    DEFAULT_ORG_UNITS = [1601]
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "tuwien-research-client/1.0"
        })
    
    def query_orgunit_publications(
        self,
        org_unit_id: int,
        from_year: int = None,
        recursive: bool = True,
        max_items: int = None
    ) -> List[Dict]:
        """
        Query publications from a specific organizational unit.
        This uses the working endpoint from your scraper.
        
        Args:
            org_unit_id: TISS organizational unit ID (e.g., 1601)
            from_year: Starting year for publications
            recursive: Include sub-units
            max_items: Maximum number of items to retrieve
        
        Returns:
            List of publication dictionaries with metadata
        """
        from_param = f"&from={from_year}" if from_year else ""
        url = (f"{self.BASE_URL}/orgunit/publications_full/tiss_id/{org_unit_id}"
               f"?count=true&recursive={str(recursive).lower()}{from_param}")
        
        results = []
        offset = 0
        
        logger.info("querying org unit %s from %s", org_unit_id, from_year or "beginning")

        while True:
            try:
                response = self.session.get(
                    f"{url}&offset={offset}",
                    timeout=30
                )

                if response.status_code != 200:
                    logger.warning("HTTP %s at offset %d", response.status_code, offset)
                    break

                data = response.json()
                total = data.get("numberOfItems", 0)
                items = data.get("items", [])

                if not items:
                    break

                logger.debug("retrieved items %d–%d of %d", offset + 1, min(offset + len(items), total), total)
                results.extend(items)

                offset += len(items)

                if max_items and len(results) >= max_items:
                    results = results[:max_items]
                    break

                if total and offset >= total:
                    break

                time.sleep(0.5)  # Be polite

            except requests.RequestException:
                logger.exception("error fetching offset %d", offset)
                break
        
        return results
    
    # PDB methods are commented out as the API requires authentication
    # def search_pdb_projects(self, search_term: str, max_results: int = 100) -> List[Dict]:
    #     """Search TISS PDB for projects."""
    #     url = f"{self.BASE_URL_PDB}/projectFullSearch/v1.0/projects"
    #     payload = {"searchTerm": search_term, "maxResults": max_results}
    #     return self.post_json(url, payload) or []

    def search_publications_by_keyword(
        self,
        keyword: str,
        org_units: List[int] = None,
        from_year: int = None,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Search for publications containing a keyword across organizational units.
        
        Args:
            keyword: Search term
            org_units: List of org unit IDs to search (default: [1601])
            from_year: Starting year
            max_results: Maximum results
        
        Returns:
            List of matching publications
        """
        if org_units is None:
            org_units = self.DEFAULT_ORG_UNITS
        
        all_pubs = []
        keyword_lower = keyword.lower()
        
        for org_id in org_units:
            pubs = self.query_orgunit_publications(
                org_id,
                from_year=from_year,
                max_items=max_results * 2  # Fetch more to filter
            )
            
            # Filter by keyword in metadata
            for pub in pubs:
                metadata = pub.get("metadata", [])
                
                # Search in title, abstract, subjects
                searchable_text = " ".join([
                    m.get("value", "") for m in metadata
                    if m.get("key") in ["dc.title", "dc.description.abstract", "dc.subject"]
                ]).lower()
                
                if keyword_lower in searchable_text:
                    all_pubs.append(pub)
                
                if len(all_pubs) >= max_results:
                    break
            
            if len(all_pubs) >= max_results:
                break
        
        return all_pubs[:max_results]
    
    def parse_metadata(self, item: Dict) -> Dict:
        """
        Parse publication metadata into structured format.
        Based on your working scraper's metadata extraction.
        """
        metadata = item.get("metadata", [])
        
        def get_meta_value(key: str, language: str = None) -> str:
            """Get single metadata value."""
            for m in metadata:
                if m.get("key") == key:
                    if language is None or m.get("language") == language:
                        return m.get("value", "")
            return ""
        
        def get_meta_values(key: str) -> List[str]:
            """Get all metadata values for a key."""
            return [m.get("value", "") for m in metadata if m.get("key") == key]
        
        # Extract organizational units
        orgunits = list({
            m.get("value", "")[:4] 
            for m in metadata 
            if m.get("key") == "tuw.publication.orgunit"
        })
        
        # Get abstracts (prefer English)
        abstract_en = get_meta_value("dc.description.abstract", "en")
        abstract_de = get_meta_value("dc.description.abstract", "de")
        abstract = abstract_en or abstract_de
        
        # Publication type
        pub_type = get_meta_value("dc.type", "en")
        
        # Extract year from date
        date_issued = get_meta_value("dc.date.issued")
        year = None
        if date_issued:
            year_match = re.search(r'\d{4}', date_issued)
            if year_match:
                year = int(year_match.group())
        
        return {
            "id": item.get("id"),
            "title": get_meta_value("dc.title"),
            "authors": get_meta_values("dc.contributor.author"),
            "abstract": abstract,
            "DOI": get_meta_value("tuw.publisher.doi") or get_meta_value("dc.identifier.doi"),
            "publication_year": year,
            "date_issued": date_issued,
            "publisher": get_meta_value("tuw.relation.publisher"),
            "type": pub_type,
            "subjects": get_meta_values("dc.subject"),
            "language": get_meta_value("dc.language.iso"),
            "orgunits": orgunits,
            "affiliations": ["TU Wien"],
            "countries": ["AT"],
            "keywords": get_meta_values("dc.subject"),
            "raw_metadata": metadata  # Keep for reference
        }
    
    # FPF/PDB API methods
    
    def get_research_areas(self) -> Optional[Dict]:
        """Fetch research areas from PDB."""
        try:
            url = f"{self.BASE_URL_PDB}/researchAreas/v2"
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            logger.exception("error fetching research areas")
            return None
    
    def get_all_research_units(self) -> List[Dict]:
        """Get all research units from FPF."""
        try:
            url = f"{self.BASE_URL_FPF}/research_unit/get_all"
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            result = r.json()
            return result if isinstance(result, list) else []
        except Exception:
            logger.exception("error fetching research units")
            return []