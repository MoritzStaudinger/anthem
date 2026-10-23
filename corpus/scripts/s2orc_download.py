"""Download Semantic Scholar / S2ORC data for ACL Anthology papers.

The Bulk Datasets API ships every dataset as a list of pre-signed URLs
pointing to gzipped JSONL shards (~5–15 GB each, hundreds total for the
full s2orc release). Storing all of it would cost ~150 GB. Instead, we
stream each shard, filter to records whose `externalIds.ACL` is set,
and write the survivors to per-paper JSONL files. The temporary .gz
shard is deleted as soon as the filter pass finishes — peak disk usage
stays at one shard.

Workflow (run in this order — each step depends on the previous one):

  python scripts/s2orc_download.py paper-ids   --output-dir <DIR>
  python scripts/s2orc_download.py papers      --output-dir <DIR>
  python scripts/s2orc_download.py s2orc       --output-dir <DIR>
  python scripts/s2orc_download.py authors     --output-dir <DIR>   # optional

Outputs (all under <DIR>):
  paper-ids/acl_to_s2.json     {ACL_id: corpus_id}
  papers/<corpus_id>.json      raw S2 papers record
  s2orc/<corpus_id>.json       raw S2ORC full-text record (subset)
  authors/<author_id>.json     raw S2 author record (optional)
  state.json                   shards already completed (for resume)
  errors.log                   any failures

Auth:  export S2_API_KEY=... (or put it in .env in the cwd). Without a key
the script still works but is throttled to ~1 req/sec and 100 req/5min.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import sys
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

S2_API_BASE = "https://api.semanticscholar.org/datasets/v1"

logger = logging.getLogger("s2orc_download")


# -------------------------- env / session --------------------------

def _load_env_file(cwd: Path) -> None:
    """Auto-load .env from cwd into os.environ (only keys not already set)."""
    env_file = cwd / ".env"
    if not env_file.is_file():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _session(api_key: Optional[str]) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=8, connect=5, read=5, status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers["User-Agent"] = "acl-processor/s2orc_download (research use)"
    if api_key:
        s.headers["x-api-key"] = api_key
    return s


# -------------------------- API helpers --------------------------

def _latest_release(session: requests.Session) -> str:
    r = session.get(f"{S2_API_BASE}/release/latest", timeout=60)
    r.raise_for_status()
    obj = r.json()
    # endpoint returns either {"release_id":...} or the release_id as a bare string,
    # depending on version. Normalize.
    if isinstance(obj, dict):
        return obj["release_id"]
    return str(obj)


def _dataset_shard_urls(
    session: requests.Session, release_id: str, dataset_name: str
) -> list[str]:
    r = session.get(
        f"{S2_API_BASE}/release/{release_id}/dataset/{dataset_name}",
        timeout=60,
    )
    if r.status_code == 404:
        raise SystemExit(
            f"dataset {dataset_name!r} not found in release {release_id}. "
            "Check spelling or release availability."
        )
    r.raise_for_status()
    obj = r.json()
    files = obj.get("files") or []
    if not files:
        raise SystemExit(
            f"dataset {dataset_name!r} listed 0 shards — has the URL expired? "
            "Signed URLs from the datasets API are short-lived (~5 min). "
            "Re-run."
        )
    return files


# -------------------------- streaming filter --------------------------

def _iter_jsonl_gz(stream: io.BufferedReader) -> Iterator[dict]:
    """Yield JSON-decoded records from a gzipped JSONL byte stream."""
    with gzip.GzipFile(fileobj=stream, mode="rb") as gz:
        for line in gz:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed


# Accept both lowercase and camelCase — bulk datasets use `corpusid`,
# some S2 releases mix in camelCase `corpusId`.
_CORPUSID_RE = re.compile(rb'"[Cc]orpus[Ii]d":\s*(\d+)')


def _iter_jsonl_gz_prefiltered(stream: io.BufferedReader,
                               wanted_cids: Set[int]) -> Iterator[dict]:
    """Yield JSON-decoded records ONLY for lines whose corpusid is in
    `wanted_cids`.

    Why: json.loads on a ~5-50 KB s2orc record costs ~0.5-1 ms. Across
    the 50-100k records per shard that's the dominant cost. With ~100 k
    ACL papers out of ~200 M S2 records, only a tiny fraction match —
    we can skip 99%+ of json.loads calls.

    Approach: regex out the FIRST corpusid in each raw line, check int
    membership in the set, only then json.loads. We search the whole
    line because S2ORC records don't have a fixed key order — `corpusid`
    can be at byte 5 in one record and byte 800 in another (after a
    large `externalids` object).

    `re.search` on the whole line short-circuits at the first match, so
    when the top-level corpusid IS near the start the cost stays low.
    Top-level corpusid is always the first occurrence in the line —
    references / cited_by arrays come later in the JSON.
    """
    with gzip.GzipFile(fileobj=stream, mode="rb") as gz:
        for raw in gz:
            if not raw:
                continue
            m = _CORPUSID_RE.search(raw)
            if not m:
                continue
            try:
                cid = int(m.group(1))
            except ValueError:
                continue
            if cid not in wanted_cids:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _download_and_filter(
    session: requests.Session,
    url: str,
    is_match: callable,
    on_match: callable,
    progress_desc: str,
) -> int:
    """Stream `url`, decompress, and call `on_match(record)` for each record
    that satisfies `is_match`. Returns count of matches.

    Note on the stream wiring: an earlier version wrapped r.raw in a
    custom io.RawIOBase subclass to drive a byte-level tqdm. That
    triggered NotImplementedError because gzip.GzipFile ultimately calls
    readinto() through BufferedReader, and RawIOBase.readinto() raises
    by default. Plain r.raw + decode_content=False is simpler and works.
    """
    print(f"{progress_desc} ...", flush=True)
    t0 = time.time()
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        # We handle gzip ourselves below; tell requests NOT to also try to
        # decompress (which would corrupt the byte stream).
        r.raw.decode_content = False
        matches = 0
        n_records = 0
        for rec in _iter_jsonl_gz(r.raw):
            n_records += 1
            if is_match(rec):
                on_match(rec)
                matches += 1
        elapsed = time.time() - t0
        print(f"{progress_desc} done: scanned {n_records:,} records, "
              f"matched {matches:,} in {elapsed:.0f}s", flush=True)
        return matches


def _download_and_filter_prefiltered(
    session: requests.Session,
    url: str,
    wanted_cid_strings: Set[str],  # kept for API symmetry; ignored internally
    is_match: callable,
    on_match: callable,
    progress_desc: str,
) -> int:
    """Same as `_download_and_filter` but skips json.loads on records
    whose corpusid isn't in the wanted set. Cuts shard wall time ~2x
    when only a tiny fraction (~0.05%) of records match."""
    # Recover the int set from the prefilter strings: each string is
    # `"corpusid":N`; pull N back out.
    wanted_cids: Set[int] = set()
    for s in wanted_cid_strings:
        try:
            wanted_cids.add(int(s.rsplit(":", 1)[1]))
        except (ValueError, IndexError):
            continue

    print(f"{progress_desc} ...", flush=True)
    t0 = time.time()
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        matches = 0
        n_decoded = 0
        for rec in _iter_jsonl_gz_prefiltered(r.raw, wanted_cids):
            n_decoded += 1
            if is_match(rec):
                on_match(rec)
                matches += 1
        elapsed = time.time() - t0
        print(f"{progress_desc} done: decoded {n_decoded:,} candidates, "
              f"matched {matches:,} in {elapsed:.0f}s", flush=True)
        return matches


# -------------------------- state / resume --------------------------

def _load_state(out_dir: Path) -> dict:
    sf = out_dir / "state.json"
    if not sf.is_file():
        return {}
    try:
        return json.loads(sf.read_text())
    except json.JSONDecodeError:
        return {}


def _save_state(out_dir: Path, state: dict) -> None:
    sf = out_dir / "state.json"
    sf.write_text(json.dumps(state, indent=2))


# -------------------------- per-dataset handlers --------------------------

def _ext(rec: dict) -> dict:
    """S2 bulk datasets use lowercase `externalids` in the JSONL. Some
    releases also expose camelCase `externalIds` — accept either."""
    return rec.get("externalids") or rec.get("externalIds") or {}


def _corpusid(rec: dict):
    """Bulk datasets use lowercase `corpusid`; graph API uses `corpusId`."""
    return rec.get("corpusid") if rec.get("corpusid") is not None else rec.get("corpusId")


def _has_acl(rec: dict) -> bool:
    return bool(_ext(rec).get("ACL"))


def cmd_paper_ids(args, session, release_id):
    """Pass 1: scan the `papers` dataset, save records with externalids.ACL
    set, and build the {ACL_id: corpusid} map.

    Historical note: this used to filter the `paper-ids` dataset, but
    recent S2 releases ship that dataset without `externalids` populated
    — running it returned 0 matches across 526M records. Filtering the
    `papers` dataset directly (which DOES carry externalids) is the right
    approach. As a bonus, we save the matched papers in the same scan,
    so there's no separate `papers` pass needed.

    Disk / bandwidth: the `papers` shards total ~50 GB (vs ~16 GB for
    paper-ids), so this is the bandwidth-heavy step. Resumable via
    state.json and the acl_to_s2.json map (both written after every shard).
    """
    out_dir = Path(args.output_dir)
    pid_dir = out_dir / "paper-ids"
    papers_dir = out_dir / "papers"
    pid_dir.mkdir(parents=True, exist_ok=True)
    papers_dir.mkdir(parents=True, exist_ok=True)

    map_path = pid_dir / "acl_to_s2.json"
    acl_map: dict[str, int] = {}
    if map_path.is_file() and not args.force:
        acl_map = json.loads(map_path.read_text())
        logger.info("loaded existing map: %d ACL papers", len(acl_map))

    state = _load_state(out_dir)
    # Key kept as "paper-ids" for backwards-compat with existing state.json
    # files, even though we're now scanning the `papers` dataset.
    done = set(state.get("paper-ids", []))

    urls = _dataset_shard_urls(session, release_id, "papers")
    logger.info("scanning `papers` dataset for ACL records: %d shards", len(urls))
    if args.dry_run:
        for u in urls[:3]:
            logger.info("  shard URL (truncated): %s ...", u[:80])
        logger.info("dry-run: stopping after listing shards.")
        return
    if args.limit_shards is not None:
        urls = urls[: args.limit_shards]
        logger.info("LIMITED to first %d shard(s) (--limit-shards)", len(urls))

    for i, url in enumerate(urls):
        if url in done and not args.force:
            continue
        try:
            def _on_match(rec: dict, _acl_map=acl_map, _tgt=papers_dir):
                acl_id = _ext(rec).get("ACL")
                cid = _corpusid(rec)
                if acl_id and cid is not None:
                    _acl_map[str(acl_id)] = int(cid)
                    safe = str(acl_id).replace("/", "_")
                    (_tgt / f"{safe}.json").write_text(
                        json.dumps(rec, ensure_ascii=False)
                    )
            n = _download_and_filter(
                session, url,
                is_match=_has_acl, on_match=_on_match,
                progress_desc=f"papers shard {i+1}/{len(urls)} (filtering for ACL)",
            )
            logger.info("shard %d/%d: +%d ACL matches (running total %d)",
                        i + 1, len(urls), n, len(acl_map))
        except Exception as e:
            logger.warning("shard %d failed: %s: %s",
                           i + 1, type(e).__name__, str(e) or repr(e))
            (out_dir / "errors.log").open("a").write(
                f"papers shard {i+1}: {type(e).__name__}: {e!r}\n"
            )
            continue
        done.add(url)
        state["paper-ids"] = sorted(done)
        _save_state(out_dir, state)
        # Persist the map after every shard so a crash doesn't lose progress.
        map_path.write_text(json.dumps(acl_map, indent=2, sort_keys=True))

    logger.info("papers scan done: %d ACL→S2 mappings → %s "
                "(papers JSONs saved under %s)",
                len(acl_map), map_path, papers_dir)


def _load_acl_map(out_dir: Path) -> dict[str, int]:
    p = out_dir / "paper-ids" / "acl_to_s2.json"
    if not p.is_file():
        raise SystemExit(
            "acl_to_s2.json not found. Run `s2orc_download.py paper-ids` first."
        )
    return json.loads(p.read_text())


def cmd_papers(args, session, release_id):
    """Pass 2: filter the `papers` dataset to records whose corpusId is in
    our ACL map; save per-paper raw S2 records."""
    out_dir = Path(args.output_dir)
    target_dir = out_dir / "papers"
    target_dir.mkdir(parents=True, exist_ok=True)

    acl_map = _load_acl_map(out_dir)
    wanted = {cid: acl_id for acl_id, cid in acl_map.items()}
    logger.info("papers: targeting %d ACL papers by corpusId", len(wanted))

    state = _load_state(out_dir)
    done = set(state.get("papers", []))

    urls = _dataset_shard_urls(session, release_id, "papers")
    logger.info("papers: %d shards", len(urls))
    if args.dry_run:
        for u in urls[:3]:
            logger.info("  shard URL (truncated): %s ...", u[:80])
        logger.info("dry-run: stopping after listing shards.")
        return
    if args.limit_shards is not None:
        urls = urls[: args.limit_shards]
        logger.info("LIMITED to first %d shard(s) (--limit-shards)", len(urls))

    for i, url in enumerate(urls):
        if url in done and not args.force:
            continue
        try:
            def _on_match(rec: dict, _wanted=wanted, _tgt=target_dir):
                cid = _corpusid(rec)
                if cid in _wanted:
                    # Save under the ACL paper_id for human-readable lookup.
                    acl_id = _wanted[cid]
                    safe = acl_id.replace("/", "_")
                    (_tgt / f"{safe}.json").write_text(
                        json.dumps(rec, ensure_ascii=False)
                    )
            n = _download_and_filter(
                session, url,
                is_match=lambda r: _corpusid(r) in wanted,
                on_match=_on_match,
                progress_desc=f"papers shard {i+1}/{len(urls)}",
            )
            logger.info("shard %d/%d: saved %d papers", i + 1, len(urls), n)
        except Exception as e:
            logger.warning("shard %d failed: %s: %s", i + 1, type(e).__name__, str(e) or repr(e))
            (out_dir / "errors.log").open("a").write(
                f"papers shard {i+1}: {e}\n"
            )
            continue
        done.add(url)
        state["papers"] = sorted(done)
        _save_state(out_dir, state)


def cmd_s2orc(args, session, release_id):
    """Pass 3: filter the `s2orc` (fulltext + structure) dataset.

    Parallelizes shard downloads up to `--workers` threads (default 4).
    Each shard streams + decodes independently — bandwidth-bound on the
    download leg, CPU-bound on JSON decode. ~3-4 threads is the sweet
    spot before the cluster's ingress bandwidth saturates.
    """
    out_dir = Path(args.output_dir)
    target_dir = out_dir / "s2orc"
    target_dir.mkdir(parents=True, exist_ok=True)

    acl_map = _load_acl_map(out_dir)
    wanted = {cid: acl_id for acl_id, cid in acl_map.items()}
    # Build a corpusid-as-string set for the prefilter — see _shard_worker.
    wanted_cid_strings = {f'"corpusid":{cid}' for cid in wanted}
    logger.info("s2orc: targeting %d ACL papers by corpusId", len(wanted))

    state = _load_state(out_dir)
    done = set(state.get("s2orc", []))

    urls = _dataset_shard_urls(session, release_id, "s2orc")
    logger.info("s2orc: %d shards", len(urls))
    if args.dry_run:
        for u in urls[:3]:
            logger.info("  shard URL (truncated): %s ...", u[:80])
        logger.info("dry-run: stopping after listing shards.")
        return
    if args.limit_shards is not None:
        urls = urls[: args.limit_shards]
        logger.info("LIMITED to first %d shard(s) (--limit-shards)", len(urls))

    todo = [(i, u) for i, u in enumerate(urls) if u not in done or args.force]
    logger.info("s2orc: %d shards to process (%d already done), workers=%d",
                len(todo), len(urls) - len(todo), args.workers)

    # Per-thread session so connection pools don't contend.
    api_key = (os.environ.get("S2_API_KEY")
               or os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))

    state_lock = threading.Lock()

    def _shard_worker(idx_url):
        i, url = idx_url
        thread_sess = _session(api_key)  # separate connection per thread
        try:
            def _on_match(rec: dict, _wanted=wanted, _tgt=target_dir):
                cid = _corpusid(rec)
                if cid in _wanted:
                    acl_id = _wanted[cid]
                    safe = acl_id.replace("/", "_")
                    (_tgt / f"{safe}.json").write_text(
                        json.dumps(rec, ensure_ascii=False)
                    )
            n = _download_and_filter_prefiltered(
                thread_sess, url,
                wanted_cid_strings=wanted_cid_strings,
                is_match=lambda r: _corpusid(r) in wanted,
                on_match=_on_match,
                progress_desc=f"s2orc shard {i+1}/{len(urls)}",
            )
            logger.info("shard %d/%d: saved %d s2orc records",
                        i + 1, len(urls), n)
        except Exception as e:
            logger.warning("shard %d failed: %s: %s",
                           i + 1, type(e).__name__, str(e) or repr(e))
            with state_lock:
                (out_dir / "errors.log").open("a").write(
                    f"s2orc shard {i+1}: {e}\n"
                )
            return
        with state_lock:
            done.add(url)
            state["s2orc"] = sorted(done)
            _save_state(out_dir, state)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_shard_worker, iu) for iu in todo]
        for fut in as_completed(futures):
            # Surface any unexpected exceptions (worker errors are
            # already logged + saved; this catches genuine bugs)
            try:
                fut.result()
            except Exception as e:
                logger.error("worker crashed: %s", e)


def cmd_authors(args, session, release_id):
    """Optional: filter authors dataset to those that appear in our papers."""
    out_dir = Path(args.output_dir)
    target_dir = out_dir / "authors"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Collect author ids from already-downloaded papers
    paper_dir = out_dir / "papers"
    if not paper_dir.is_dir():
        raise SystemExit("Run `papers` first (need it to know which authors to save).")
    wanted_ids = set()
    for jp in paper_dir.glob("*.json"):
        try:
            rec = json.loads(jp.read_text())
            for a in (rec.get("authors") or []):
                aid = a.get("authorId")
                if aid:
                    wanted_ids.add(str(aid))
        except Exception:
            continue
    logger.info("authors: targeting %d unique authorIds", len(wanted_ids))

    state = _load_state(out_dir)
    done = set(state.get("authors", []))
    urls = _dataset_shard_urls(session, release_id, "authors")
    if args.dry_run:
        for u in urls[:3]:
            logger.info("  shard URL (truncated): %s ...", u[:80])
        logger.info("dry-run: stopping after listing shards.")
        return
    if args.limit_shards is not None:
        urls = urls[: args.limit_shards]
        logger.info("LIMITED to first %d shard(s) (--limit-shards)", len(urls))

    for i, url in enumerate(urls):
        if url in done and not args.force:
            continue
        try:
            def _on_match(rec: dict, _w=wanted_ids, _tgt=target_dir):
                aid = str(rec.get("authorid") or rec.get("authorId") or "")
                if aid in _w:
                    (_tgt / f"{aid}.json").write_text(
                        json.dumps(rec, ensure_ascii=False)
                    )
            _ = _download_and_filter(
                session, url,
                is_match=lambda r: str(r.get("authorid") or r.get("authorId") or "") in wanted_ids,
                on_match=_on_match,
                progress_desc=f"authors shard {i+1}/{len(urls)}",
            )
        except Exception as e:
            logger.warning("shard %d failed: %s: %s", i + 1, type(e).__name__, str(e) or repr(e))
            (out_dir / "errors.log").open("a").write(
                f"authors shard {i+1}: {e}\n"
            )
            continue
        done.add(url)
        state["authors"] = sorted(done)
        _save_state(out_dir, state)


# -------------------------- main --------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("paper-ids", "papers", "s2orc", "authors"),
                   help="Which dataset shard to download/filter. Run in order.")
    p.add_argument("--output-dir", required=True,
                   help="Where to store the filtered records (and state.json).")
    p.add_argument("--release-id", default=None,
                   help="Optional release pin (e.g., 2024-12-17). Defaults "
                        "to the latest release.")
    p.add_argument("--force", action="store_true",
                   help="Re-download shards already marked complete in state.json.")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel shard downloads (s2orc step only). "
                        "Default 4 — beyond ~6 the cluster's ingress "
                        "bandwidth saturates and adding more threads "
                        "doesn't help. Set to 1 for the old sequential "
                        "behavior.")
    p.add_argument("--limit-shards", type=int, default=None,
                   help="Stop after processing N shards. Use for smoke testing "
                        "(1 paper-ids shard finishes in ~5–10 min).")
    p.add_argument("--dry-run", action="store_true",
                   help="List the dataset's shard URLs and exit. No download. "
                        "Useful to confirm the API endpoints + your key work.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    _load_env_file(Path.cwd())
    api_key = os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        logger.warning(
            "no S2_API_KEY in env — falling back to public quota (1 req/sec). "
            "Register a key at semanticscholar.org/product/api/key and "
            "add `S2_API_KEY=...` to your .env.",
        )

    session = _session(api_key)
    release_id = args.release_id or _latest_release(session)
    logger.info("using release_id=%s", release_id)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    handlers = {
        "paper-ids": cmd_paper_ids,
        "papers":    cmd_papers,
        "s2orc":     cmd_s2orc,
        "authors":   cmd_authors,
    }
    handlers[args.step](args, session, release_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
