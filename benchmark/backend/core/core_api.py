import os
import requests
import urllib.parse

api_key = os.getenv("CORE_API_KEY", "jXou1LrHsDMPZ2e3cU0OFTyRCJdkYq4w")

def query_api(search_url, query, limit, scroll=False, scrollId=None):
    headers = {"Authorization": "Bearer " + api_key.strip()}
    if not scrollId and scroll:
        url = f"{search_url}?q={query}&limit={limit}&scroll=true"
    elif scrollId:
        url = f"{search_url}?q={query}&limit={limit}&scrollId={scrollId}"
    else:
        url = f"{search_url}?q={query}&limit={limit}"

    response = requests.get(url, headers=headers)
    if response.status_code in [429, 500, 503]:
        return query_api(search_url, query, limit, scroll, scrollId)
    return response.json(), response.elapsed.total_seconds()

def search_works(search_query, limit=10):
    response, _ = query_api(
        "https://api.core.ac.uk/v3/search/works",
        urllib.parse.quote(f"{search_query} and _exists_:description"),
        limit=limit,
    )
    results = []
    seen_titles = set()
    for hit in response.get("results", []):
        title = hit.get("title")
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        results.append({
            "url": f"https://core.ac.uk/works/{hit.get('id', 'unknown')}",
            "abstract": (hit.get("abstract") or "")[:800],
            "title": title,
            "authors": [author.get("name") for author in hit.get("authors", [])],
            "fulltext": hit.get("fullText", ""),
            "doi": hit.get("doi"),
            "published_date": hit.get("publishedDate", ""),
            "updated_date": hit.get("updatedDate", "")
        })
    return results
