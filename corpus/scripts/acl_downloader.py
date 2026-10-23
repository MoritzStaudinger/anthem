# scripts/acl_downloader.py
#
# Walks the cloned acl-org/acl-anthology repo's data/xml/ directory
# directly with stdlib ElementTree — no `acl_anthology` library
# dependency. This way newly-introduced schema elements (e.g., the
# paper-level <month>/<year> overrides added April 2026) don't break
# the download just because the library wasn't updated yet.
import os
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


def get_paper_metadata(anthology_repo: str):
    """Extract [(paper_id, pdf_url), ...] from the anthology repo's XML
    files. Library-free — uses stdlib XML parsing and silently ignores
    elements/attributes the schema added after this script was written.

    PDF URLs follow the canonical pattern
        https://aclanthology.org/<paper_anthology_id>.pdf
    where <paper_anthology_id> is read from each paper's <url> child
    (e.g., '2024.acl-long.1' or 'P18-1001'). Papers without a <url> child
    have no downloadable PDF and are skipped.
    """
    repo = Path(anthology_repo)
    xml_dir = repo / "data" / "xml"
    if not xml_dir.is_dir():
        xml_dir = repo
    xml_files = sorted(xml_dir.glob("*.xml"))
    print(f"Initializing ACL Anthology from {xml_dir} "
          f"({len(xml_files)} XML files)...")

    pairs = []
    bad_files = 0
    for xml_file in tqdm(xml_files, desc="Scanning XML"):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as e:
            bad_files += 1
            tqdm.write(f"  WARN: unreadable {xml_file.name}: {e}")
            continue
        root = tree.getroot()
        for volume in root.iter("volume"):
            for paper in volume.findall("paper"):
                url_el = paper.find("url")
                if url_el is None:
                    continue
                anth_id = (url_el.text or "").strip()
                if not anth_id:
                    continue
                pdf_url = f"https://aclanthology.org/{anth_id}.pdf"
                pairs.append((anth_id, pdf_url))

    if bad_files:
        print(f"  ({bad_files} XML file(s) skipped due to parse errors)")
    print(f"Scanned {len(pairs):,} papers with PDFs across "
          f"{len(xml_files) - bad_files} XML files.")
    return pairs

def _requests_session(verify_ssl: bool, retries: int, backoff: float, timeout: int):
    s = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "ACL Anthology Downloader (research use)"})
    # store options on session
    s.verify = verify_ssl
    s.request_timeout = timeout
    return s

def download_pdfs(papers, output_dir, log_file, max_workers, verify_ssl, retries, backoff, timeout):
    """Download PDFs to output_dir, named <paper_id>.pdf with parallel workers."""
    print(f"\nFound {len(papers)} PDFs to potentially download.")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Optional: quiet SSL warnings when verify_ssl is False
    if not verify_ssl:
        requests.packages.urllib3.disable_warnings(
            requests.packages.urllib3.exceptions.InsecureRequestWarning
        )

    session = _requests_session(verify_ssl, retries, backoff, timeout)

    def _download_one(paper_tuple):
        paper_id, url = paper_tuple
        filename = f"{paper_id}.pdf"
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            return ("skipped", paper_id, url, None)
        try:
            with session.get(url, stream=True, timeout=session.request_timeout) as r:
                if r.status_code == 200 and r.headers.get("content-type", "").lower().startswith("application/pdf"):
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    return ("downloaded", paper_id, url, None)
                elif r.status_code == 200:
                    # some servers don't label properly; still save
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    return ("downloaded", paper_id, url, "non-pdf content-type")
                else:
                    return ("failed", paper_id, url, f"HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            return ("failed", paper_id, url, f"RequestException: {e}")
        except Exception as e:
            return ("failed", paper_id, url, f"Unexpected error: {e}")

    failures = 0
    with open(log_file, "a") as flog:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_download_one, p) for p in papers]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDFs"):
                status, pid, url, reason = fut.result()
                if status == "failed":
                    failures += 1
                    msg = f"Failed {pid} from {url}: {reason}"
                    flog.write(msg + "\n")
                # we stay silent on "skipped" to keep logs lean

    print("\nDownload process completed.")
    if failures:
        print(f"Failures logged: {failures} -> {os.path.abspath(log_file)}")

def main():
    parser = argparse.ArgumentParser(
        description="Download ACL Anthology PDFs (parallel) with unique IDs for filenames."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save PDFs.")
    parser.add_argument("--log_file", type=str, default="logs/acl_download_errors.log",
                        help="File to log errors.")
    parser.add_argument("--anthology_repo", type=str, required=True,
                        help="Path to local ACL Anthology repository clone.")
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=30, help="Per request timeout (seconds).")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retry attempts.")
    parser.add_argument("--retry_backoff", type=float, default=0.5, help="Exponential backoff factor.")
    parser.add_argument("--verify_ssl", action="store_true", help="Enable SSL verification (default off).")

    args = parser.parse_args()

    try:
        papers = get_paper_metadata(args.anthology_repo)
        download_pdfs(
            papers,
            args.output_dir,
            args.log_file,
            args.max_workers,
            args.verify_ssl,
            args.retries,
            args.retry_backoff,
            args.timeout,
        )
        print(f"PDFs saved in '{os.path.abspath(args.output_dir)}'")
        print(f"Log file: '{os.path.abspath(args.log_file)}'")
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Check that --anthology_repo points to a valid local clone.")

if __name__ == "__main__":
    main()
