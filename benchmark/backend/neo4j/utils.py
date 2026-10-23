from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal

from neo4j import GraphDatabase


# ---- Minimal "instances" you can use in your app ----

@dataclass
class Author:
    name: str
    id: Optional[str] = None  # optional if your graph has an id field


@dataclass
class Publication:
    id: Optional[str]
    title: Optional[str]
    year: Optional[int]
    abstract: Optional[str]


@dataclass
class PublicationWithAuthors:
    publication: Publication
    authors: List[Author]


# ---- Minimal Neo4j reader ----

class Neo4jReader:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "12345678")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self.driver.close()

    def search_publications_with_authors(self, text: str, limit: int = 20, year_start=None, year_end=None):
        cypher = """
        MATCH (p:Publication)<-[:AUTHORED]-(a:Person)
        WHERE (
            toLower(coalesce(p.title,'')) CONTAINS toLower($text)
            OR toLower(coalesce(p.abstract,'')) CONTAINS toLower($text)
        )
        AND ($year_start IS NULL OR toInteger(p.year) >= $year_start)
        AND ($year_end   IS NULL OR toInteger(p.year) <= $year_end)
        WITH p, collect(DISTINCT a) AS authors
        RETURN
          p.id       AS id,
          p.title    AS title,
          p.year     AS year,
          p.abstract AS abstract,
          [x IN authors | {
            name: trim(coalesce(x.first_name,'') + ' ' + coalesce(x.last_name,'')),
            id: coalesce(x.id, null)
          }] AS authors
        ORDER BY coalesce(p.year, 0) DESC
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                cypher,
                text=text,
                limit=limit,
                year_start=year_start,
                year_end=year_end,
            ).data()

        out: List[PublicationWithAuthors] = []
        for r in rows:
            pub = Publication(
                id=r.get("id"),
                title=r.get("title"),
                year=r.get("year"),
                abstract=r.get("abstract"),
            )
            authors = [Author(name=a.get("name"), id=a.get("id")) for a in (r.get("authors") or [])]
            out.append(PublicationWithAuthors(publication=pub, authors=authors))

        return out

    def search_publications_by_research_unit(
        self,
        unit_text: str,
        *,
        year: Optional[int] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        years: Optional[List[int]] = None,
        limit: int = 200,
        affiliation: Literal["direct_or_indirect", "direct_only"] = "direct_or_indirect",
        text: Optional[str] = None,  # optional title/abstract filter
        return_authors: bool = True,
    ):
        """
        Find publications authored by people affiliated (directly or indirectly) with a ResearchUnit,
        within a flexible timeframe.

        Timeframe options (choose one style):
          - year=2024
          - year_start=2020, year_end=2024
          - years=[2022, 2023, 2024]

        unit_text: substring match on ru.name (case-insensitive).
        affiliation:
          - "direct_or_indirect": uses :DIR_OR_INDIR_AFF_UNIT
          - "direct_only": uses :DIR_AFF_UNIT (if your graph has it; adjust if different)

        text: optional substring match over publication title/abstract.
        """

        # Pick relationship type for affiliation
        aff_rel = "DIR_OR_INDIR_AFF_UNIT" if affiliation == "direct_or_indirect" else "DIR_AFF_UNIT"

        cypher = f"""
        MATCH (ru:ResearchUnit)<-[:{aff_rel}]-(p:Person)-[:AUTHORED]->(pu:Publication)<-[autFull:AUTHORED]-(pa:Person)
        WHERE lower(coalesce(ru.name,'')) CONTAINS lower($unit_text)

          // timeframe filter
          AND (
                ($years IS NOT NULL AND toInteger(pu.year) IN $years)
             OR ($years IS NULL AND $year IS NOT NULL AND toInteger(pu.year) = $year)
             OR ($years IS NULL AND $year IS NULL
                 AND ($year_start IS NULL OR toInteger(pu.year) >= $year_start)
                 AND ($year_end   IS NULL OR toInteger(pu.year) <= $year_end)
                )
          )

          // optional text filter
          AND (
                $text IS NULL
             OR lower(coalesce(pu.title,'')) CONTAINS lower($text)
             OR lower(coalesce(pu.abstract,'')) CONTAINS lower($text)
          )
        """

        if return_authors:
            cypher += """
                      WITH pu \
                         , collect(DISTINCT pa) AS authors
                          RETURN
                          pu.id AS id \
                         , pu.title AS title \
                         , pu.year AS year \
                         , pu.abstract AS abstract \
                         , [a IN authors | {
                          name : trim (coalesce (a.first_name \
                         , '') + ' ' + coalesce (a.last_name \
                         , '')) \
                         , id: coalesce (a.id \
                         , null)
                          }] AS authors
                      ORDER BY coalesce (toInteger(pu.year), 0) DESC
                          LIMIT $limit \
                      """
        else:
            cypher += """
            RETURN
              pu.id       AS id,
              pu.title    AS title,
              pu.year     AS year,
              pu.abstract AS abstract
            ORDER BY coalesce(toInteger(pu.year), 0) DESC
            LIMIT $limit
            """

        params = dict(
            unit_text=unit_text,
            text=text,
            limit=limit,
            year=year,
            year_start=year_start,
            year_end=year_end,
            years=years,
        )

        with self.driver.session(database=self.database) as session:
            rows = session.run(cypher, **params).data()

        # Reuse your existing output model style
        out: List[PublicationWithAuthors] = []
        for r in rows:
            pub = Publication(
                id=r.get("id"),
                title=r.get("title"),
                year=r.get("year"),
                abstract=r.get("abstract"),
            )
            authors = [Author(name=a["name"], id=a.get("id")) for a in (r.get("authors") or [])]
            out.append(PublicationWithAuthors(publication=pub, authors=authors))

        return out

def search_paper_graph(
            graph_client,
            query: str,
            year_start: Optional[int] = None,
            year_end: Optional[int] = None,
            samples: int = 100,
    ) -> List[dict]:
        """
        Graph-backed replacement for search_paper(...).
        Returns list[dict] compatible with downstream pipeline.
        """

        # If you add year filters in Cypher, pass them through instead.
        # For now we do minimal retrieval by text + post-filtering on year.
        reader = Neo4jReader()
        pubs = reader.search_publications_with_authors(text=query, limit=samples)

        works = [pub_with_authors_to_work_dict(p) for p in pubs]

        return works

from typing import List, Dict, Any, Optional

def pub_with_authors_to_work_dict(pwa) -> Dict[str, Any]:
    pub = pwa.publication
    authors = pwa.authors or []

    authorships = []
    for a in authors:
        authorships.append({
            "author": {
                "display_name": getattr(a, "name", None),
                "id": getattr(a, "id", None),
            },
            "institutions": [],
        })

    year = getattr(pub, "year", None)
    if year is not None:
        year = int(year)

    title = getattr(pub, "title", None)
    abstract = getattr(pub, "abstract", None)

    # DOI mapping
    pub_id = getattr(pub, "id", None)
    doi = None
    if pub_id:
        s = str(pub_id).strip()
        # If someone accidentally stored a full URL, normalize back to DOI
        s = s.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi.org/", "")
        doi = s or None

    return {
        "title": title,
        "display_name": title,
        "abstract": abstract,
        "publication_year": year,
        "authorships": authorships,
        "all_authorships": authorships,
        "doi": doi,                              # <-- now set
        "id": f"https://doi.org/{doi}" if doi else None,  # optional but useful
        "is_oa": None,
        "language": "en",
    }

def search_paper_graph_by_unit(
    *,
    reader: Neo4jReader,
    unit_text: str,
    year: Optional[int] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    years: Optional[List[int]] = None,
    samples: int = 100,
    text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    pubs = reader.search_publications_by_research_unit(
        unit_text=unit_text,
        year=year,
        year_start=year_start,
        year_end=year_end,
        years=years,
        limit=samples,
        text=text,
        return_authors=True,
    )
    return [pub_with_authors_to_work_dict(p) for p in pubs]
