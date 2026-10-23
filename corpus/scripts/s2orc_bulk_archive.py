"""Bulk download every shard of every S2ORC dataset, save as-is to disk.

Unlike s2orc_download.py which streams + filters to ACL records (peak
disk usage = 1 shard), this saves the FULL gzipped JSONL archives. Use
when you want the complete S2 corpus available locally for re-filtering,
non-ACL analyses, or as a permanent snapshot.

Disk budget (approximate, post-2026-05 release):
    paper-ids   ~16 GB    (30 shards × ~500 MB)
    papers      ~50 GB
    s2orc       ~150 GB   ← the big one (full-text records)
    authors     ~10 GB
    abstracts   ~30 GB    (optional)
    ----
    total       ~250 GB compressed

Time budget: bandwidth-bound. At ~50 MB/s sustained you're looking at
~90 min total. Cluster head nodes may be slower.

Robustness:
    - Uses HTTP `Range:` requests to RESUME interrupted downloads.
    - On any failure, the partial shard stays on disk; re-run resumes it.
    - Signed URLs expire after ~5 min. If a download starts after expiry,
      we re-fetch the URL list once and try again.
    - Verifies file size after each download.

Usage:
    python scripts/s2orc_bulk_archive.py \\
        --datasets paper-ids papers s2orc authors \\
        --output-dir /share/moritz.staudinger/acl_data/s2orc_full_archive

    # Or just one dataset (e.g., for testing):
    python scripts/s2orc_bulk_archive.py --datasets paper-ids \\
        --output-dir /share/test/s2orc

Auth: S2_API_KEY in .env (free, register at
       https://www.semanticscholar.org/product/api/key)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

S2_API_BASE = "https://api.semanticscholar.org/datasets/v1"

logger = logging.getLogger("s2orc_bulk")


def _load_env(cwd: Path) -> None:
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


def _api_session(api_key: Optional[str]) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "acl-processor/s2orc_bulk_archive"
    if api_key:
        s.headers["x-api-key"] = api_key
    return s


def _latest_release(session: requests.Session) -> str:
    r = session.get(f"{S2_API_BASE}/release/latest", timeout=60)
    r.raise_for_status()
    obj = r.json()
    return obj["release_id"] if isinstance(obj, dict) else str(obj)


def _list_shards(session: requests.Session, release_id: str, dataset: str) -> List[str]:
    r = session.get(f"{S2_API_BASE}/release/{release_id}/dataset/{dataset}",
                    timeout=60)
    r.raise_for_status()
    return r.json().get("files") or []


def _shard_filename(url: str, idx: int) -> str:
    """Derive a stable filename for a shard. S2 URLs typically end in
    /shard-XX.gz; we keep that suffix if present, else number sequentially."""
    path = urlparse(url).path
    base = Path(path).name
    if base.endswith(".gz") and len(base) > 3:
        return base
    return f"shard-{idx:03d}.jsonl.gz"


def _download_shard(url: str, dst: Path, timeout: int = 600,
                    chunk_size: int = 1 << 20) -> bool:
    """Resumable single-shard download. Returns True on success.

    Uses Range: bytes=<offset>- to continue from a partial file. If the
    server gives 200 (full response) instead of 206 (partial), we
    truncate and start over.
    """
    headers = {}
    existing = dst.stat().st_size if dst.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    try:
        # Don't use a session here — bypass any retry adapter. We do our
        # own retries at the orchestration level so we can refresh URLs.
        with requests.get(url, headers=headers, stream=True,
                          timeout=timeout) as r:
            if r.status_code == 416:
                # "Requested range not satisfiable" — file is already complete
                logger.info("  ✓ %s already complete (%d bytes)", dst.name, existing)
                return True
            if r.status_code not in (200, 206):
                logger.warning("  HTTP %d for %s", r.status_code, dst.name)
                return False
            # Resume only valid if server returned 206; on 200 we restart
            mode = "ab" if r.status_code == 206 else "wb"
            if mode == "wb" and existing:
                logger.info("  server didn't honour Range; restarting %s", dst.name)
                existing = 0
            written = existing
            with dst.open(mode) as fh:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
                        written += len(chunk)
            content_length = r.headers.get("content-length")
            if content_length:
                expected_total = int(content_length) + existing
                if written != expected_total:
                    logger.warning("  size mismatch %s: got %d expected %d",
                                   dst.name, written, expected_total)
                    return False
            logger.info("  ✓ %s (%.1f MB)", dst.name, written / (1 << 20))
            return True
    except Exception as e:
        logger.warning("  download %s failed: %s: %s",
                       dst.name, type(e).__name__, str(e) or repr(e))
        return False


def _archive_dataset(session: requests.Session, release_id: str,
                     dataset: str, dst_dir: Path,
                     max_url_refresh: int = 2) -> None:
    """Archive every shard of one dataset. Handles signed-URL expiry by
    re-fetching the URL list up to `max_url_refresh` times."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    refresh_count = 0
    urls = _list_shards(session, release_id, dataset)
    logger.info("%s: %d shards to archive into %s", dataset, len(urls), dst_dir)

    failed_idx: List[int] = []
    for i, url in enumerate(urls):
        fname = _shard_filename(url, i)
        dst = dst_dir / fname
        if dst.exists() and dst.stat().st_size > 0:
            # Already started — try to resume. _download_shard handles 416.
            pass
        ok = _download_shard(url, dst)
        if not ok:
            failed_idx.append(i)

    # If anything failed, try once with refreshed URLs
    while failed_idx and refresh_count < max_url_refresh:
        refresh_count += 1
        logger.info("%s: retrying %d failed shard(s) with refreshed URLs (%d/%d)",
                    dataset, len(failed_idx), refresh_count, max_url_refresh)
        urls = _list_shards(session, release_id, dataset)
        still_failed: List[int] = []
        for i in failed_idx:
            if i >= len(urls):
                logger.warning("shard index %d out of range after refresh", i)
                still_failed.append(i)
                continue
            url = urls[i]
            fname = _shard_filename(url, i)
            dst = dst_dir / fname
            ok = _download_shard(url, dst)
            if not ok:
                still_failed.append(i)
        failed_idx = still_failed

    if failed_idx:
        logger.warning("%s: %d shard(s) FAILED after refreshes: %s",
                       dataset, len(failed_idx), failed_idx[:10])
    else:
        logger.info("%s: all shards archived successfully", dataset)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["paper-ids", "papers", "s2orc"],
                    help="Which datasets to archive. Skip 'authors' / 'abstracts' "
                         "unless you need them — saves ~40 GB.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Root output dir. One subdir per dataset.")
    ap.add_argument("--release-id", default=None,
                    help="Optional pinned release (e.g. 2026-05-12). "
                         "Defaults to whatever 'release/latest' returns.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    _load_env(Path.cwd())
    api_key = (os.environ.get("S2_API_KEY")
               or os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    if not api_key:
        raise SystemExit("S2_API_KEY not set in env or .env")

    session = _api_session(api_key)
    release_id = args.release_id or _latest_release(session)
    logger.info("release_id=%s, output_root=%s", release_id, args.output_dir)

    # Persist a manifest of what we're downloading
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release_id": release_id,
        "datasets": args.datasets,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    t0 = time.time()
    for ds in args.datasets:
        logger.info("==================== %s ====================", ds)
        try:
            _archive_dataset(session, release_id, ds, args.output_dir / ds)
        except Exception as e:
            logger.error("dataset %s: %s: %s", ds, type(e).__name__,
                         str(e) or repr(e))
    logger.info("done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
