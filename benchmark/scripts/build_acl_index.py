"""Build the local ACL anthology index.

One-off CLI: pair every `*.md` in --text-dir with its same-stem `*.json`
in --meta-dir, embed via sentence-transformers MiniLM, persist a
llama-index VectorStoreIndex under --persist-dir.

Run this once when the corpus changes; the agent's search_local_acl_papers
tool will lazy-load the persisted index on first use.

Examples
--------
    python scripts/build_acl_index.py \\
        --text-dir data/acl_corpus/texts \\
        --meta-dir data/acl_corpus/meta

    # custom output dir
    python scripts/build_acl_index.py \\
        --text-dir ./acl/texts --meta-dir ./acl/meta \\
        --persist-dir data/local_index_acl_2025

The persist-dir defaults to data/local_index_acl which is what
search_local_acl_papers looks at.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/build_acl_index.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag.local_acl_indexing import (  # noqa: E402
    DEFAULT_ACL_PERSIST_DIR,
    build_acl_index,
)
from backend.rag.scico_indexing import (  # noqa: E402
    ABSTRACT_ONLY,
    DEFAULT_SCICO_ABSTRACTS_PERSIST_DIR,
    DEFAULT_SCICO_FULLTEXT_PERSIST_DIR,
    DEFAULT_SCICO_ROOT,
    PREFER_FULLTEXT,
    build_scico_index,
    build_scico_index_resumable,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a local ACL / SciCo VectorStoreIndex.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--scico-root",
        help=(
            f"Shortcut: point at a SciCo dataset root (containing "
            f"affiliations/, pymu_llm/, ground_truth.csv). "
            f"Default: {DEFAULT_SCICO_ROOT}. Mutually exclusive with --text-dir/--meta-dir."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("abstracts", "fulltext"),
        default="abstracts",
        help=(
            "SciCo only. `abstracts`: index abstract per paper "
            f"(fast, ideal for institution/topic queries; default persist: "
            f"{DEFAULT_SCICO_ABSTRACTS_PERSIST_DIR}). "
            "`fulltext`: index pymu_llm markdown when present, fall back to "
            f"abstract (default persist: {DEFAULT_SCICO_FULLTEXT_PERSIST_DIR})."
        ),
    )
    p.add_argument(
        "--text-dir",
        help="Directory containing paper full-text files (default glob *.md). "
             "Use with --meta-dir for the legacy text+meta layout.",
    )
    p.add_argument(
        "--meta-dir",
        help="Directory containing paper metadata JSON (one file per paper, "
             "stem matches the text file).",
    )
    p.add_argument(
        "--persist-dir",
        default=None,
        help=(
            "Where to persist the built index. For --scico-root, defaults to "
            f"{DEFAULT_SCICO_ABSTRACTS_PERSIST_DIR} (abstracts) or "
            f"{DEFAULT_SCICO_FULLTEXT_PERSIST_DIR} (fulltext) depending on "
            f"--mode. For --text-dir/--meta-dir, defaults to "
            f"{DEFAULT_ACL_PERSIST_DIR}."
        ),
    )
    p.add_argument(
        "--text-glob",
        default="*.md",
        help="Glob for text files within --text-dir (default: *.md). Ignored for --scico-root.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "SciCo only. Process papers in batches, checkpoint after each, "
            "skip paper_ids already in <persist-dir>/processed.txt on "
            "restart. Recommended for the 100k corpus."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Resumable mode only — papers per checkpoint batch (default 500).",
    )
    p.add_argument(
        "--limit-papers",
        type=int,
        default=None,
        help="Smoke test: process only the first N papers. Useful to "
             "validate the enriched-corpus pipeline before paying for a "
             "full ~30 min build.",
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Resumable mode only — skip intermediate persists between "
             "batches. ~halves wall time on a full 121k-paper build at "
             "the cost of crash-recovery (a mid-build crash means "
             "starting over). Final persist still happens at the end.",
    )
    p.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-document progress bar.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    legacy = bool(args.text_dir or args.meta_dir)
    scico = args.scico_root is not None

    # If neither flag is given, default to the SciCo layout in this repo.
    if not legacy and not scico:
        args.scico_root = DEFAULT_SCICO_ROOT
        scico = True

    if legacy and scico:
        p.error("Use either --scico-root OR --text-dir/--meta-dir, not both.")
    if legacy and not (args.text_dir and args.meta_dir):
        p.error("--text-dir requires --meta-dir (and vice versa).")

    if scico:
        text_strategy = ABSTRACT_ONLY if args.mode == "abstracts" else PREFER_FULLTEXT
        if args.persist_dir:
            persist_dir = args.persist_dir
        else:
            persist_dir = (
                DEFAULT_SCICO_ABSTRACTS_PERSIST_DIR
                if args.mode == "abstracts"
                else DEFAULT_SCICO_FULLTEXT_PERSIST_DIR
            )
        if args.resume:
            build_scico_index_resumable(
                scico_root=args.scico_root,
                persist_dir=persist_dir,
                text_strategy=text_strategy,
                batch_size=args.batch_size,
                show_progress=not args.quiet,
                limit_papers=args.limit_papers,
                no_checkpoint=args.no_checkpoint,
            )
            print(f"\n✓ SciCo {args.mode} index persisted (resumable) to {persist_dir}")
        else:
            build_scico_index(
                scico_root=args.scico_root,
                persist_dir=persist_dir,
                text_strategy=text_strategy,
                show_progress=not args.quiet,
            )
            print(f"\n✓ SciCo {args.mode} index persisted to {persist_dir}")
    else:
        persist_dir = args.persist_dir or DEFAULT_ACL_PERSIST_DIR
        build_acl_index(
            text_dir=args.text_dir,
            meta_dir=args.meta_dir,
            persist_dir=persist_dir,
            text_glob=args.text_glob,
            show_progress=not args.quiet,
        )
        print(f"\n✓ ACL index persisted to {persist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
