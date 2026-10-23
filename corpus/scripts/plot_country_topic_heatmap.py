"""Country × Topic heatmap from topics_by_country.csv.

Companion to scripts/plot_treemap.py. Pivots the long-form CSV to a
matrix, row-normalises so each country's row sums to 100% across the
displayed topics (i.e. cell value = % of that country's *displayed*
papers tagged with that topic), and renders a labelled heatmap.

The row-normalisation lets you compare countries' *specialisations*
without scale bias from how many papers each country publishes.

Usage:
    python scripts/plot_country_topic_heatmap.py \\
        --csv  reports/aggregates/topics_by_country.csv \\
        --top-countries 12 --top-topics 18 \\
        --exclude "linguistics,natural language processing,natural languages,language model,computational linguistics,computer science,university,semantics,syntactics" \\
        --out  reports/treemaps/country_topic_heatmap_all_years.pdf \\
        --title "ACL Anthology 1952-2025 — country specialisation by topic"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("country_topic_heatmap")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path,
                    help="topics_by_country.csv (long-form).")
    ap.add_argument("--country-col", default="country_label")
    ap.add_argument("--topic-col", default="cso_label")
    ap.add_argument("--value-col", default="paper_count")
    ap.add_argument("--top-countries", type=int, default=12)
    ap.add_argument("--top-topics", type=int, default=18)
    ap.add_argument("--exclude", default="",
                    help="Comma-separated topic labels to drop before "
                         "ranking (umbrella topics).")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--annotate", action="store_true", default=True,
                    help="Write the percentage inside each cell.")
    ap.add_argument("--no-annotate", dest="annotate", action="store_false")
    ap.add_argument("--figsize", default="14,9")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(args.csv)
    for col in (args.country_col, args.topic_col, args.value_col):
        if col not in df.columns:
            raise SystemExit(f"--csv missing column '{col}' "
                             f"(have: {list(df.columns)})")

    excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}
    df = df[~df[args.topic_col].isin(excluded)]
    df = df[df[args.topic_col].notna()
            & (df[args.topic_col].astype(str).str.len() > 0)]
    df = df[df[args.country_col].notna()
            & (df[args.country_col].astype(str).str.len() > 0)]
    df = df[df[args.value_col].fillna(0) > 0]

    countries = (df.groupby(args.country_col)[args.value_col]
                   .sum().sort_values(ascending=False)
                   .head(args.top_countries).index.tolist())
    topics = (df.groupby(args.topic_col)[args.value_col]
                .sum().sort_values(ascending=False)
                .head(args.top_topics).index.tolist())
    logger.info("countries: %s", ", ".join(map(str, countries)))
    logger.info("topics: %s", ", ".join(map(str, topics)))

    # Pivot to country (row) × topic (col), filling missing pairs with 0
    pivot = (df[df[args.country_col].isin(countries)
              & df[args.topic_col].isin(topics)]
             .pivot_table(index=args.country_col, columns=args.topic_col,
                          values=args.value_col, aggfunc="sum",
                          fill_value=0)
             .reindex(index=countries, columns=topics, fill_value=0))

    # Row-normalise (each country's row sums to 100% across displayed topics)
    row_totals = pivot.sum(axis=1).replace(0, 1)
    pct = pivot.div(row_totals, axis=0) * 100.0

    fig_w, fig_h = (float(s) for s in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    im = ax.imshow(pct.values, aspect="auto", cmap=args.cmap,
                   vmin=0, vmax=max(20.0, pct.values.max()))

    ax.set_xticks(np.arange(len(topics)))
    ax.set_xticklabels(topics, rotation=40, ha="right",
                       fontsize=9.5)
    ax.set_yticks(np.arange(len(countries)))
    ax.set_yticklabels(countries, fontsize=10)
    ax.set_xlabel("CSO topic", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    ax.tick_params(top=False, bottom=True, left=True, right=False)

    # Faint white grid lines for cell separation
    ax.set_xticks(np.arange(-.5, len(topics), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(countries), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", length=0)

    if args.annotate:
        # Pick text color per cell by luminance against the chosen cmap
        cmap = plt.get_cmap(args.cmap)
        max_v = max(20.0, pct.values.max())
        for i in range(len(countries)):
            for j in range(len(topics)):
                v = pct.values[i, j]
                if v < 0.5:        # don't crowd near-zero cells
                    continue
                rgba = cmap(v / max_v)
                lum = (0.2126 * rgba[0] + 0.7152 * rgba[1]
                       + 0.0722 * rgba[2])
                tc = "white" if lum < 0.5 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color=tc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("% of country's displayed papers", rotation=270,
                   labelpad=14, fontsize=10)

    if args.title:
        ax.set_title(args.title, fontsize=12, fontweight="bold", pad=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
