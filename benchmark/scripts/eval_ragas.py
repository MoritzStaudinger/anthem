"""RAGAS additive judge pass over saved eval_scico answers.

Reads $answers_dir/answers.jsonl (written by eval_scico_answer.sbatch or
eval_scico.sbatch with --generate-only) and writes per-row RAGAS scores
to $answers_dir/ragas_<judge_slug>.jsonl. The output joins back to other
reports on `id`.

This is purely additive: it does NOT re-run the agent, modify
answers.jsonl, or interact with the existing in-house judge pipeline
(_judge_free_form_grounded). Run it alongside eval_scico_judge.sbatch to
get RAGAS metrics as an external reference, while keeping your custom
phantom_citation_rate / affiliation_correctness / precision_recall_mrr
metrics unchanged.

Metrics:
  - Faithfulness        : answer claims grounded in retrieved contexts
  - ResponseRelevancy   : answer addresses the question (embeddings-based)

Judge swap is the whole point: pass any model id the existing
configure_llm dispatcher knows (OpenAI / Anthropic / Google / Aqueduct /
OpenRouter). Run the script once per judge to build a per-judge file.

CAVEAT — context completeness:
  Evidence records (`records` in answers.jsonl) are written by
  agent_tools._record_evidence, which intentionally drops abstracts to
  keep the critic prompt small. RAGAS Faithfulness atomizes the answer
  into claims and checks each against contexts, so without abstracts it
  can only verify "this paper exists with title X / author Y / year Z" —
  any methodology/finding claim will look unsupported and faithfulness
  scores will be pessimistic. If full faithfulness matters for the
  paper, enrich agent_tools._record_evidence with `abstract` (small,
  surgical change — adds tokens to the critic prompt) and re-run
  generation. For the current pass we use title + authors + year +
  affiliations as the per-paper context block.

Usage:
    python scripts/eval_ragas.py \\
        --answers-dir reports/eval_scico/agent_deepseek_v4_flash \\
        --judge claude-opus-4-7

    # Multi-judge: re-run per judge; outputs land in distinct files.
    for j in claude-opus-4-7 gpt-5 qwen/qwen3.5-397b-a17b; do
        python scripts/eval_ragas.py \\
            --answers-dir reports/eval_scico/agent_deepseek_v4_flash \\
            --judge "$j"
    done

Resumable: re-running picks up where the previous run stopped (skips IDs
already present in the output file).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Project root on sys.path so `import backend.*` works regardless of cwd
# (mirrors the pattern used in scripts/eval_scico.py and friends).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    """Filesystem-safe slug from a model id (e.g. 'qwen/qwen3.5-397b-a17b'
    -> 'qwen_qwen3_5_397b_a17b')."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")


def _load_dotenv(path: Path) -> int:
    """Minimal .env loader (python-dotenv is not a project dependency).

    Parses KEY=VALUE lines and writes them into os.environ WITHOUT overriding
    any variable already exported in the shell (so `export OPENAI_API_KEY=...`
    still wins). Returns the number of keys loaded; no-ops if the file is
    absent. Lets the user drop the judge API key into .env and just run."""
    if not path.exists():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
            loaded += 1
    return loaded


def _load_testset(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return id -> testset row. Needed because answers.jsonl doesn't
    store the original query."""
    out: Dict[str, Dict[str, Any]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("id")
            if rid:
                out[rid] = row
    return out


def _row_is_complete(rec: Dict[str, Any]) -> bool:
    """A row is 'complete' (don't retry) when:
      - it was legitimately skipped upstream (no query / no contexts / agent
        error), OR
      - both RAGAS metrics scored as real numbers.
    Rows with any null metric or _outer_error are treated as failed and
    eligible for retry under --retry-failed.
    """
    if rec.get("skipped_reason"):
        return True
    f = rec.get("faithfulness")
    r = rec.get("answer_relevancy")
    return isinstance(f, (int, float)) and isinstance(r, (int, float))


def _load_done_ids(out_path: Path, retry_failed: bool = False) -> Set[str]:
    """Resume support: ids whose RAGAS scores are already saved.

    When retry_failed=True, ids whose row is NOT complete (per
    _row_is_complete) are excluded — they get re-processed."""
    done: Set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" not in rec:
                continue
            if retry_failed and not _row_is_complete(rec):
                continue
            done.add(rec["id"])
    return done


def _purge_failed_rows(out_path: Path) -> Tuple[int, int]:
    """Rewrite out_path keeping only rows where _row_is_complete is True.
    Returns (kept, dropped). No-op when file is missing."""
    if not out_path.exists():
        return 0, 0
    kept_lines: List[str] = []
    dropped = 0
    with out_path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if _row_is_complete(rec):
                kept_lines.append(line if line.endswith("\n") else line + "\n")
            else:
                dropped += 1
    with out_path.open("w", encoding="utf-8") as fh:
        fh.writelines(kept_lines)
    return len(kept_lines), dropped


def _records_to_contexts(records: List[Dict[str, Any]]) -> List[str]:
    """Turn the agent's slim evidence records into RAGAS contexts.

    One context string per paper. Includes whatever metadata is on the
    record — abstract if present (rich), otherwise title+authors+year+
    affiliations (thin; see module docstring caveat)."""
    contexts: List[str] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        title = (rec.get("title") or "").strip()
        abstract = (rec.get("abstract") or "").strip()
        authors = rec.get("authors") or []
        if isinstance(authors, list):
            authors_str = ", ".join(a for a in authors if a)
        else:
            authors_str = str(authors)
        year = rec.get("publication_year") or rec.get("year") or ""
        affs = rec.get("affiliations") or []
        if isinstance(affs, list):
            affs_str = "; ".join(a for a in affs if a)
        else:
            affs_str = str(affs)
        doi = rec.get("DOI") or rec.get("doi") or rec.get("url") or ""

        parts = []
        if title:
            parts.append(f"Title: {title}")
        if authors_str:
            parts.append(f"Authors: {authors_str}")
        if year:
            parts.append(f"Year: {year}")
        if affs_str:
            parts.append(f"Affiliations: {affs_str}")
        if doi:
            parts.append(f"DOI: {doi}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if parts:
            contexts.append("\n".join(parts))
    return contexts


def _context_source(records: List[Dict[str, Any]]) -> str:
    """Label which fields the contexts ended up using — paper-table fact."""
    has_abstract = any(
        (isinstance(r, dict) and (r.get("abstract") or "").strip())
        for r in (records or [])
    )
    return "with_abstracts" if has_abstract else "metadata_only"


async def _score_sample(sample, metrics) -> Dict[str, Any]:
    """Score one sample across all metrics. Per-metric exceptions are
    isolated so one bad metric doesn't drop the row."""
    out: Dict[str, Any] = {}
    for m in metrics:
        name = getattr(m, "name", None) or type(m).__name__.lower()
        try:
            score = await m.single_turn_ascore(sample)
            out[name] = float(score) if score is not None else None
            out[f"{name}_error"] = None
        except Exception as e:
            out[name] = None
            out[f"{name}_error"] = f"{type(e).__name__}: {e}"
            logger.warning("metric %s failed: %s", name, e)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--answers-dir", type=Path, required=True,
                   help="Directory containing answers.jsonl")
    p.add_argument("--judge", required=True,
                   help="Judge model id (routed via backend.rag.rag_utils.configure_llm)")
    p.add_argument("--testset", type=Path,
                   default=Path("data/eval/scico_testset.jsonl"),
                   help="Testset path (needed to recover query per id)")
    p.add_argument("--out-name", default=None,
                   help="Output filename (default: ragas_<judge_slug>.jsonl)")
    p.add_argument("--limit", type=int, default=None,
                   help="Score at most N new rows (smoke test).")
    p.add_argument("--no-resume", action="store_true",
                   help="Overwrite the output file instead of appending.")
    p.add_argument("--retry-failed", action="store_true",
                   help="Drop rows with null metrics / _outer_error from "
                        "the existing output file and re-score them. Rows "
                        "with real scores or legitimate skipped_reason are "
                        "kept untouched. Safe to re-run any number of times.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    n_env = _load_dotenv(ROOT / ".env")
    if n_env:
        print(f"# loaded {n_env} var(s) from {ROOT / '.env'} "
              "(shell exports take precedence)", flush=True)

    answers_path = args.answers_dir / "answers.jsonl"
    if not answers_path.exists():
        print(f"ERROR: {answers_path} not found", file=sys.stderr)
        return 1
    if not args.testset.exists():
        print(f"ERROR: testset {args.testset} not found", file=sys.stderr)
        return 1

    # Late imports so --help works without ragas/llama-index installed.
    from llama_index.core import Settings  # noqa: E402
    from backend.rag.rag_utils import configure_llm  # noqa: E402

    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    except ImportError as e:
        print(f"ERROR: ragas not installed ({e}). "
              "pip install 'ragas>=0.2'", file=sys.stderr)
        return 1

    print(f"# loading testset {args.testset}", flush=True)
    testset = _load_testset(args.testset)
    print(f"# testset has {len(testset)} rows", flush=True)

    judge_slug = _slug(args.judge)
    out_name = args.out_name or f"ragas_{judge_slug}.jsonl"
    out_path = args.answers_dir / out_name

    print(f"# configuring judge LLM: {args.judge}", flush=True)
    configure_llm(args.judge)

    # Token instrumentation: attach a counter to the global callback manager
    # AND to the concrete llm/embed objects RAGAS will call, so we can report
    # real judge token/cost per answer. The ragas_*.jsonl files never logged
    # this — it's the missing piece for the cost-accounting rebuttal number
    # (marginal cost of an additional judge).
    from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
    from backend.eval.metrics import estimate_cost_usd
    token_counter = TokenCountingHandler()
    Settings.callback_manager = CallbackManager([token_counter])
    for _obj in (Settings.llm, Settings.embed_model):
        try:
            _obj.callback_manager = Settings.callback_manager
        except Exception:
            pass

    judge_llm = LlamaIndexLLMWrapper(Settings.llm)
    judge_embed = LlamaIndexEmbeddingsWrapper(Settings.embed_model)

    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_embed),
    ]

    if args.no_resume and args.retry_failed:
        print("ERROR: --no-resume and --retry-failed are mutually exclusive",
              file=sys.stderr)
        return 1

    if args.retry_failed:
        kept, dropped = _purge_failed_rows(out_path)
        print(f"# --retry-failed: kept {kept} complete rows, "
              f"dropped {dropped} failed rows from {out_path.name}",
              flush=True)

    done: Set[str] = (
        set() if args.no_resume
        else _load_done_ids(out_path, retry_failed=args.retry_failed)
    )
    if done:
        print(f"# resume: {len(done)} rows already scored in {out_path.name}",
              flush=True)

    mode = "w" if args.no_resume else "a"
    n_scored = 0
    n_skipped_done = 0
    n_skipped_empty = 0
    n_failed = 0
    n_missing_query = 0

    with answers_path.open() as fin, out_path.open(mode, encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = row.get("id")
            if rid is None:
                continue
            if rid in done:
                n_skipped_done += 1
                continue
            if args.limit is not None and n_scored >= args.limit:
                break

            # Skip rows the agent failed on
            if row.get("error"):
                rec = {
                    "id": rid,
                    "judge": args.judge,
                    "skipped_reason": f"upstream agent error: {row.get('error')}",
                    "faithfulness": None,
                    "answer_relevancy": None,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_skipped_empty += 1
                continue

            ts_row = testset.get(rid) or {}
            question = ts_row.get("query") or ""
            if not question:
                # Without a question we can't score; record and move on.
                rec = {
                    "id": rid,
                    "judge": args.judge,
                    "skipped_reason": "no query in testset",
                    "faithfulness": None,
                    "answer_relevancy": None,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_missing_query += 1
                continue

            answer = (row.get("summary_text") or "").strip()
            records = row.get("records") or []
            contexts = _records_to_contexts(records)
            ctx_src = _context_source(records)

            if not answer or not contexts:
                rec = {
                    "id": rid,
                    "judge": args.judge,
                    "skipped_reason": ("empty answer" if not answer else "no contexts"),
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "n_contexts": len(contexts),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_skipped_empty += 1
                continue

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )

            token_counter.reset_counts()
            try:
                scores = asyncio.run(_score_sample(sample, metrics))
            except Exception as e:
                # Outer failure (rare — only if asyncio itself blows up)
                scores = {"faithfulness": None, "answer_relevancy": None,
                          "_outer_error": f"{type(e).__name__}: {e}"}
                n_failed += 1

            # Real judge token/cost for THIS answer (the missing cost number).
            j_prompt = int(token_counter.prompt_llm_token_count)
            j_completion = int(token_counter.completion_llm_token_count)
            j_embed = int(token_counter.total_embedding_token_count)
            j_cost = estimate_cost_usd(args.judge, j_prompt, j_completion)
            rec = {
                "id": rid,
                "judge": args.judge,
                "n_contexts": len(contexts),
                "context_source": ctx_src,  # 'with_abstracts' or 'metadata_only'
                "judge_prompt_tokens": j_prompt,
                "judge_completion_tokens": j_completion,
                "judge_embedding_tokens": j_embed,
                "judge_total_tokens": j_prompt + j_completion,
                "judge_cost_usd": j_cost,
                **scores,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_scored += 1

            if n_scored % 10 == 0:
                print(f"# scored {n_scored} rows ({n_failed} failed so far)",
                      flush=True)

    print(
        f"# done — scored={n_scored} skipped_done={n_skipped_done} "
        f"skipped_empty={n_skipped_empty} missing_query={n_missing_query} "
        f"failed={n_failed}",
        flush=True,
    )
    print(f"# output: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
