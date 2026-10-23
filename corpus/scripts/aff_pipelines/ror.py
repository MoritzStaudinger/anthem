"""Shared ROR resolver — wraps scripts/ror_link_annotation.py's HTTP
matcher in a class with a disk cache, callable from any pipeline.

Single instance shared across all pipelines so cache hits compound.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ror_link_annotation import (  # noqa: E402
    ror_match, _flatten_one_match, load_cache, save_cache,
)

logger = logging.getLogger("ror_resolver")


_ROR_API_BASE = "https://api.ror.org/v2/organizations"


class RorResolver:
    """Resolve a raw affiliation string -> {ror_id, clean_institution_name,
    country_code}. Returns None when nothing clears `min_confidence`.

    Two persistent caches:
      cache_path        -> {affiliation_string: matches_list} (matcher endpoint)
      hierarchy_cache_path -> {ror_id: {parent: [ror_ids], child: [ror_ids],
                                       related: [ror_ids]}} (org-by-id endpoint)

    The hierarchy cache supports the hierarchy-aware ROR scoring track in
    the comparison harness: a predicted ROR id can be considered a match
    when it shares a parent/child link with the gold id (typical case:
    Stanford-CS-Department vs Stanford-University).
    """

    def __init__(self, cache_path: Optional[Path] = None,
                 hierarchy_cache_path: Optional[Path] = None,
                 min_confidence: float = 0.7, max_retries: int = 3,
                 timeout: int = 15):
        self.cache_path = Path(cache_path) if cache_path else None
        self.hierarchy_cache_path = Path(hierarchy_cache_path) if hierarchy_cache_path else None
        self.min_confidence = min_confidence
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache: Dict[str, dict] = {}
        self._hier_cache: Dict[str, Dict[str, list]] = {}
        if self.cache_path and self.cache_path.is_file():
            self._cache = load_cache(self.cache_path)
            logger.info("loaded %d cached ROR lookups from %s",
                        len(self._cache), self.cache_path)
        if self.hierarchy_cache_path and self.hierarchy_cache_path.is_file():
            try:
                import json
                self._hier_cache = json.loads(self.hierarchy_cache_path.read_text())
                logger.info("loaded %d cached ROR hierarchies from %s",
                            len(self._hier_cache), self.hierarchy_cache_path)
            except (json.JSONDecodeError, OSError):
                pass
        # Reusable session — the ROR endpoint rate-limits per-connection,
        # so sharing one session across the whole run is the right move.
        self._session = requests.Session()
        retry = Retry(total=max_retries, backoff_factor=0.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=["GET"])
        self._session.mount("https://", HTTPAdapter(max_retries=retry))
        self._new_lookups = 0
        self._new_hier_lookups = 0

    def resolve(self, affiliation: str) -> Optional[dict]:
        """Return {ror_id, clean_institution_name, country_code} or None."""
        if not affiliation or not affiliation.strip():
            return None
        key = affiliation.strip()
        if key in self._cache:
            return self._best_match(self._cache[key])
        # Cache miss — hit the API
        try:
            matches = ror_match(key, self._session,
                                retries=self.max_retries, timeout=self.timeout)
        except Exception as e:
            logger.warning("ROR lookup failed for %r: %s", key[:80], e)
            return None
        self._cache[key] = matches
        self._new_lookups += 1
        # Periodic save so a crash doesn't lose lookup work
        if self.cache_path and self._new_lookups % 25 == 0:
            self.save()
        return self._best_match(matches)

    def _best_match(self, matches) -> Optional[dict]:
        """Pick the highest-confidence match above threshold."""
        if not matches:
            return None
        best = None
        best_score = -1.0
        for m in matches:
            flat = _flatten_one_match(m)
            try:
                score = float(flat.get("ror_confidence") or 0)
            except ValueError:
                score = 0
            if score >= self.min_confidence and score > best_score:
                best = flat
                best_score = score
        if not best:
            return None
        return {
            "ror_id": best.get("ror_id") or None,
            "clean_institution_name": best.get("ror_name") or None,
            "country_code": best.get("ror_country_code") or None,
        }

    def save(self) -> None:
        if self.cache_path:
            save_cache(self.cache_path, self._cache)
            logger.debug("saved ROR cache (%d entries) -> %s",
                         len(self._cache), self.cache_path)
        if self.hierarchy_cache_path:
            import json
            self.hierarchy_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.hierarchy_cache_path.write_text(json.dumps(self._hier_cache))
            logger.debug("saved ROR hierarchy cache (%d entries) -> %s",
                         len(self._hier_cache), self.hierarchy_cache_path)

    # Callable interface — lets pipelines use the same `ror_resolver(s)`
    # idiom regardless of whether they got an RorResolver or a bare function.
    def __call__(self, affiliation: str) -> Optional[dict]:
        return self.resolve(affiliation)

    # -------------------- hierarchy lookups --------------------

    @staticmethod
    def _strip_ror_url(rid: str) -> str:
        """ror_id stored as either URL or bare. Hierarchy cache uses bare."""
        if not rid:
            return ""
        rid = rid.strip().rstrip("/")
        return rid.split("/")[-1].lower()

    def get_relationships(self, ror_id: str) -> Dict[str, list]:
        """Fetch + cache parent/child/related ROR ids for a given org.

        Returns {"parent": [bare_ids], "child": [bare_ids], "related": [bare_ids]}.
        Always returns a dict (empty lists when no relationships) so callers
        can index without None-checks.
        """
        bare = self._strip_ror_url(ror_id)
        if not bare:
            return {"parent": [], "child": [], "related": []}
        if bare in self._hier_cache:
            return self._hier_cache[bare]
        try:
            r = self._session.get(f"{_ROR_API_BASE}/{bare}", timeout=self.timeout)
            if r.status_code != 200:
                logger.warning("ROR org-by-id %s -> HTTP %d", bare, r.status_code)
                self._hier_cache[bare] = {"parent": [], "child": [], "related": []}
                return self._hier_cache[bare]
            org = r.json()
            relationships = org.get("relationships") or []
            buckets: Dict[str, list] = {"parent": [], "child": [], "related": []}
            for rel in relationships:
                rtype = (rel.get("type") or "").lower()
                rid = self._strip_ror_url(rel.get("id") or "")
                if rid and rtype in buckets:
                    buckets[rtype].append(rid)
            self._hier_cache[bare] = buckets
            self._new_hier_lookups += 1
            if self.hierarchy_cache_path and self._new_hier_lookups % 25 == 0:
                self.save()
            return buckets
        except requests.RequestException as e:
            logger.warning("ROR org-by-id %s failed: %s", bare, e)
            return {"parent": [], "child": [], "related": []}

    def get_related_set(self, ror_id: str, include_related: bool = False) -> set:
        """Set of ROR ids 'equivalent' to `ror_id` for hierarchy-aware matching.

        Default: {self, parents, children}. The `related` ROR relationship
        type is excluded by default because it's a much looser link
        ('related to' is used for e.g. successor orgs, sister labs) that
        bloats false-positive matches. Set include_related=True if you
        want a more permissive expansion.
        """
        bare = self._strip_ror_url(ror_id)
        if not bare:
            return set()
        h = self.get_relationships(bare)
        out = {bare}
        out.update(h.get("parent") or [])
        out.update(h.get("child") or [])
        if include_related:
            out.update(h.get("related") or [])
        return out

    def warm_hierarchy(self, ror_ids) -> None:
        """Bulk pre-fetch relationships for a list of ROR ids.
        Useful at the start of the comparison run so scoring doesn't
        interleave API calls with logic."""
        unique = {self._strip_ror_url(r) for r in ror_ids if r}
        unique.discard("")
        missing = [r for r in unique if r not in self._hier_cache]
        if not missing:
            return
        logger.info("warming hierarchy cache: %d new ROR ids to fetch", len(missing))
        for r in missing:
            self.get_relationships(r)
        self.save()
