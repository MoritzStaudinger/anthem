"""Plot per-year corpus trends with mean ± std bands.

Reads reports/yearly_stats.csv (produced by aggregate_yearly_stats.py)
and emits three vector PDFs:

  1. <out-dir>/papers_per_year.pdf       — bar chart, n papers / year
  2. <out-dir>/authors_per_paper.pdf     — line + ±1 std band, optional IQR
  3. <out-dir>/institutions_per_paper.pdf — line + ±1 std band, optional IQR

Usage:
    python scripts/plot_yearly_stats.py \\
        --csv  reports/yearly_stats.csv \\
        --out-dir reports/yearly_plots/ \\
        --year-min 1980 --year-max 2025
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("plot_yearly_stats")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=Path("reports/yearly_stats.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("reports/yearly_plots"))
    ap.add_argument("--year-min", type=int, default=None,
                    help="Crop the x-axis lower bound (default: full range).")
    ap.add_argument("--year-max", type=int, default=None)
    ap.add_argument("--show-iqr", action="store_true",
                    help="Overlay the inter-quartile range (p25–p75) as a "
                         "lighter band. Often more informative than ±1 std "
                         "because per-paper author/institution counts are "
                         "right-skewed.")
    ap.add_argument("--figsize-papers", default="8,4")
    ap.add_argument("--figsize-perpaper", default="8,4")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(args.csv)
    if args.year_min is not None:
        df = df[df["year"] >= args.year_min]
    if args.year_max is not None:
        df = df[df["year"] <= args.year_max]
    df = df.sort_values("year").reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------- 1) papers per year (bar chart) -------
    fw, fh = (float(s) for s in args.figsize_papers.split(","))
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=args.dpi)
    ax.bar(df["year"], df["n_papers"],
           color="#3a6ea5", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    out1 = args.out_dir / "papers_per_year.pdf"
    fig.tight_layout()
    fig.savefig(out1, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    logger.info("wrote %s", out1)
    plt.close(fig)

    # ------- 2) authors per paper (line + ±1 std band) -------
    def _trend(metric: str, ylabel: str, out_path: Path, color: str):
        fw_, fh_ = (float(s) for s in args.figsize_perpaper.split(","))
        fig, ax = plt.subplots(figsize=(fw_, fh_), dpi=args.dpi)
        mean = df[f"{metric}_mean"]
        std = df[f"{metric}_std"]
        # ±1 std band — clip lower bound at 0 (counts can't go negative)
        lo = (mean - std).clip(lower=0)
        hi = mean + std
        ax.fill_between(df["year"], lo, hi, color=color, alpha=0.20,
                        label="±1 std")
        if args.show_iqr:
            ax.fill_between(df["year"],
                            df[f"{metric}_p25"],
                            df[f"{metric}_p75"],
                            color=color, alpha=0.35, label="IQR (p25–p75)")
        ax.plot(df["year"], mean, color=color, linewidth=1.6,
                marker="o", markersize=2.5, label="mean")
        ax.plot(df["year"], df[f"{metric}_median"], color=color,
                linewidth=0.9, linestyle="--", alpha=0.8, label="median")
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                    facecolor="white")
        logger.info("wrote %s", out_path)
        plt.close(fig)

    _trend("authors",
           "Authors per paper",
           args.out_dir / "authors_per_paper.pdf",
           color="#c0392b")
    _trend("insts",
           "Distinct institutions per paper",
           args.out_dir / "institutions_per_paper.pdf",
           color="#27ae60")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
