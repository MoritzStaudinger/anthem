import logging
import time
import urllib
from collections import Counter, defaultdict

import pyalex
import requests
from pyalex import Works, Institutions, config
from requests import RequestException
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.utils.utils import remove_duplicates_by_doi_dict

logger = logging.getLogger(__name__)

pyalex.config.email = "moritz.staudinger@tuwien.ac.at"
pyalex.config.max_retries = 0
pyalex.config.retry_backoff_factor = 0.1
pyalex.config.retry_http_codes = [429, 500, 503]

def get_most_relevant_institutions_for_query(query, year_start=None, year_end=None, country_code=None, limit=10):
    from_date = f"{year_start}-01-01" if year_start else "1970-01-01"
    to_date = f"{year_end}-12-31" if year_end else "2025-12-31"

    # Initialize the base query with filters
    works_query = Works().search(query).filter(
        is_oa=True,
        language='en',
        has_doi=True,
        from_publication_date=from_date,
        to_publication_date=to_date
    )
    # Apply country code filter if provided
    if country_code:
        works_query = works_query.filter(institutions={"country_code": country_code.upper()})

    def is_in_country(ror_id, target_code):
        _, inst = Institutions().get(ror_id)  # Unpack status code and response
        return inst.get('country_code', '').upper() == target_code.upper()



    # Group by institution ROR ID and retrieve results
    grouped_results = works_query.group_by("institutions.ror").get()
    filtered_grouped_results = [res for res in grouped_results if is_in_country(res['key'], country_code)]

    # Limit the number of institutions returned
    return filtered_grouped_results[:limit]

def get_most_relevant_institutions_for_queries(queries, year_start=None, year_end=None, country_code=None, limit=10, exclude_ror=None):
    from_date = f"{year_start}-01-01" if year_start else "1970-01-01"
    to_date = f"{year_end}-12-31" if year_end else "2025-12-31"

    institution_counts = {}

    for query in queries:
        works_query = Works().search(query).filter(
            is_oa=True,
            language='en',
            has_doi=True,
            from_publication_date=from_date,
            to_publication_date=to_date
        )

        if country_code:
            works_query = works_query.filter(institutions={"country_code": country_code.upper()})

        grouped_results = works_query.group_by("institutions.ror").get()

        for result in grouped_results:
            ror_id = result['key']
            if exclude_ror and ror_id == exclude_ror:
                continue
            count = result['count']
            institution_counts[ror_id] = institution_counts.get(ror_id, 0) + count
        time.sleep(2)

    # Sort by count descending and return top results
    sorted_results = sorted(institution_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:limit]

def structure_data(data):
    structured_data = []
    for entry in data:
        authors = [author['author']['display_name'] for author in entry['authorships']]
        affiliations = (list(set([el['display_name'] for affiliation in entry['all_authorships'] for el in affiliation['institutions']])))
        countries = (list(set([el['country_code'] for affiliation in entry['authorships'] for el in affiliation['institutions']])))
        keywords = [el['display_name'] for el in  entry['keywords']]

        # Fall back to the OpenAlex Work URL when the paper has no DOI; that
        # way every reference can be linked even if upstream metadata is thin.
        doi = entry.get('doi')
        openalex_url = entry.get('id')  # e.g. "https://openalex.org/W4400024729"
        url = doi or openalex_url

        structured_data.append({
            "title": entry['title'],
            "authors": authors,
            "abstract": entry['abstract'],
            "DOI": doi,
            "openalex_url": openalex_url,
            "url": url,  # canonical link: DOI when present, OpenAlex ID otherwise
            "publication_year": entry['publication_year'],
            "cited_by": entry['cited_by_count'],
            "language": entry['language'],
            "affiliations": affiliations,
            "countries": countries,
            "keywords": keywords,
            "topic": entry['primary_topic']['display_name'] if entry['primary_topic'] is not None else "No topic available",
            "pdf_path": entry['primary_location']['pdf_url'] if entry['primary_location'] is not None and entry['primary_location']['pdf_url'] else "No PDF available"
        })
    return structured_data

def search_paper(query, institution_id=None, year_start=None, year_end=None, samples=10000, filter_authors=False, open_access_only=True):
    """
    Sample papers by topic with pagination.
    Optionally filters authors to only include those affiliated with a given institution ROR ID.
    """
    from_date = f"{year_start}-01-01" if year_start else "1970-01-01"
    to_date = f"{year_end}-12-31" if year_end else "2025-12-31"

    works_query = Works().search(query).filter(
        is_oa=open_access_only,
        language='en',
        has_doi=True,
        from_publication_date=from_date,
        to_publication_date=to_date
    )

    if institution_id:
        works_query = works_query.filter(institutions={"ror": institution_id})

    # Pagination
    per_page = 200
    pager = works_query.paginate(per_page=per_page)

    # Collect and optionally filter results
    all_results = []
    for page in pager:
        for work in page:
            work['all_authorships'] = work.get("authorships", [])
            if filter_authors and institution_id:
                filtered_authorships = [
                    auth for auth in work.get("authorships", [])
                    if any(inst.get("ror") == institution_id for inst in auth.get("institutions", []))
                ]
                if not filtered_authorships:
                    continue  # Skip work with no matching authors
                work["authorships"] = filtered_authorships
            all_results.append(work)
            if len(all_results) >= samples:
                break
        if len(all_results) >= samples:
            break

    all_results = all_results[:samples]
    formatted_data = structure_data(all_results)
    return formatted_data

def get_ror_id(university_name):
    """
    Retrieves the ROR ID for a given university name using PyAlex.

    Args:
        university_name (str): The name of the university.

    Returns:
        str or None: The ROR ID if a match is found; otherwise, None.
    """
    try:
        # Search for institutions matching the university name
        results = Institutions().search(university_name).get()

        # Iterate through the results to find the best match
        #for institution in results:
        #    display_name = institution.get("display_name", "").lower()
        #    if university_name.lower() in display_name:
        #        return institution.get("ror")
        #return None
        logger.info("Found %d institutions matching %r", len(results), university_name)
        for r in results:
            logger.debug(" - %s (%s)", r.get("display_name"), r.get("ror"))
        return results[0].get("ror") if results else None
    except Exception:
        logger.exception("ROR lookup failed for %r", university_name)
        return None

def get_university_name_from_ror(ror_id):
    """
    Retrieves the university name for a given ROR ID using PyAlex.

    Args:
        ror_id (str): The ROR ID of the university.

    Returns:
        str or None: The university name if found; otherwise, None.
    """
    try:
        # Fetch institution by ROR ID
        institution = Institutions()[f"https://ror.org/{ror_id}"]
        return institution.get("display_name")
    except Exception:
        logger.exception("name lookup failed for ROR %r", ror_id)
        return None



def get_top_cited_papers(papers, top_k=5):
    sorted_papers = sorted(papers, key=lambda p: p.get("cited_by", 0), reverse=True)
    return sorted_papers[:top_k]


def extract_topics_from_papers(papers, num_clusters=1):
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [(p.get("title") or "") + " " + (p.get("abstract") or "") for p in papers]

    # Filter out empty texts
    texts = [text.strip() for text in texts if text.strip()]
    if not texts:
        return ["No valid text data available for topic extraction."]

    embeddings = sentence_model.encode(texts)
    if embeddings.size == 0:
        return ["Embedding generation failed; no data to cluster."]

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    labels = kmeans.labels_

    topics = []
    for cluster_id in range(num_clusters):
        cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]
        if not cluster_texts:
            topics.append("No texts in this cluster.")
            continue
        tfidf = TfidfVectorizer(ngram_range=(3, 5), max_features=3, stop_words="english").fit(cluster_texts)
        top_keywords = tfidf.get_feature_names_out()
        topics.append(", ".join(top_keywords))

    return topics

def get_institution_profile(topic, ror_id, queries=None, year_start=None, year_end=None):
    if not ror_id:
        return {
            "Institution": get_university_name_from_ror(ror_id),
            "Publications": -1,
            "Top Researchers": "N/A"
        }

    papers = search_paper(
        query=topic,  # Empty means fetch all available
        institution_id=ror_id,
        year_start=year_start,
        year_end=year_end,
        filter_authors=True,
        samples=10000,
    )
    for query in (queries or []):
        paper_query= search_paper(
            query=query,
            institution_id=ror_id,
            year_start=year_start,
            year_end=year_end,
            filter_authors=True,
            samples=10000,
        )
        papers.extend(paper_query)
    papers = remove_duplicates_by_doi_dict(papers) # Remove duplicates
    logger.info("institution profile: %d papers after dedup", len(papers))



    # Count publications
    publication_count = len(papers)

    # Count authors
    all_authors = []
    all_collaborators = []
    for paper in papers:
        all_authors.extend(paper.get("authors", []))
        all_collaborators.extend(paper.get("affiliations", []))
    top_authors = Counter(all_authors).most_common(5)
    top_authors_str = "\n\r".join(name for name, _ in top_authors) if top_authors else "N/A"

    top_affiliations = Counter(all_collaborators).most_common(15)
    top_affiliations = top_affiliations[1:]  # Skip the first one which is usually the institution itself

    topic_counts = Counter(p.get("topic") for p in papers if p.get("topic"))
    top_topics = topic_counts.most_common(15)

    citation_distribution = [int(p.get("cited_by") or 0) for p in papers]

    top_papers = get_top_cited_papers(papers)
    formatted = [
        {
            "Title": p["title"],
            "Citations": p["cited_by"],
            "involved_institution_authors": p['authors'],
            "DOI": p.get("DOI", "N/A")
        }
        for p in top_papers
    ]

    return {
        "Institution": get_university_name_from_ror(ror_id),
        "Publications": publication_count,
        "PublicationsPerYear": group_by_year(papers),
        "Year Start": year_start,
        "Year End": year_end,
        "Researchers": top_authors_str,
        "ResearchersWithPublication": publications_by_author(papers, target_authors=top_authors),
        # Heatmap data: include every author (not just top_authors) so the
        # frontend can list them all on the Y axis.
        "ResearchersWithPublicationsPerYear": publications_by_author_per_year(papers, target_authors=None),
        "Collaborator_Universities": top_affiliations,
        "TopicDistribution": top_topics,
        "CitationDistribution": citation_distribution,
        "Topics": extract_topics_from_papers(papers),
        "Top Cited Papers": formatted,
    }


def group_by_year(papers):
    """

    Group papers by publication year and count the number per year.
    Expects structured papers (e.g., from `structure_data` output).
    """
    year_counts = defaultdict(int)
    for paper in papers:
        date_str = str(paper.get("publication_date") or paper.get("publication_year"))
        if date_str:
            try:
                year = int(date_str[:4])
                year_counts[year] += 1
            except ValueError:
                continue
    return dict(sorted(year_counts.items()))


def publications_by_author(papers, target_authors=None):
    """
    Returns a dict of authors → {year → count}, optionally filtering to a given list of author names.

    Args:
        papers: list of structured papers (with authorship info)
        target_authors: list of author names to include (case-insensitive match). If None, include all.
    """
    author_publication_counts = Counter()
    target_authors_name = [name for name, _ in target_authors]
    for paper in papers:
        for author in paper.get('authors', []):
            if author in target_authors_name:
                author_publication_counts[author] += 1
    return author_publication_counts

def publications_by_author_per_year(papers, target_authors=None):
    """Return [{Author, Year, Publications, Titles}] rows for the heatmap.

    If target_authors is None, every author in the paper list is included
    (not just the top N) — useful for the frontend Authors×Years heatmap.
    Titles is a newline-joined list of paper titles for that (author, year)
    cell, capped at 8 to keep tooltips reasonable.
    """
    name_filter = None
    if target_authors:
        name_filter = {name for name, _ in target_authors}

    # author -> year (str) -> list of titles
    author_year_titles = defaultdict(lambda: defaultdict(list))

    for paper in papers:
        year = str(paper.get('publication_year', 'Unknown'))
        title = (paper.get('title') or '').strip()
        for author in paper.get('authors', []):
            if name_filter is not None and author not in name_filter:
                continue
            author_year_titles[author][year].append(title)

    rows = []
    title_cap = 8
    for author, year_titles in author_year_titles.items():
        for year, titles in year_titles.items():
            shown = titles[:title_cap]
            joined = "\n".join(f"• {t}" for t in shown if t)
            if len(titles) > title_cap:
                joined += f"\n…and {len(titles) - title_cap} more"
            rows.append({
                "Author": author,
                "Year": year,
                "Publications": len(titles),
                "Titles": joined or "(no titles recorded)",
            })
    return rows
