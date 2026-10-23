"""3-level nested squarified treemap: Country → Institution → Topic.

Layers:
  1. Outer    — top countries, sized by total papers per country
                (from institutions_by_country.csv).
  2. Middle   — within each country, top K institutions, sized by
                paper count.
  3. Innermost — within each institution, top L topics, sized by
                topic-tag count (from institutions_by_topic.csv).

Because each paper has multiple CSO topics, an institution's topic
counts sum to MORE than its paper count. The innermost squarify uses
those topic-tag counts to size the sub-cells — which gives the right
"what fraction of this institution's research is on X vs Y" reading,
not "how many papers".

Defaults are conservative (9 countries × 4 institutions × 3 topics ≈
108 leaf cells) so labels stay legible. For an exploratory poster
figure, bump --top-countries / --top-institutions / --top-topics.

Usage:
    python scripts/plot_treemap_3level.py \\
        --insts-csv  reports/aggregates/institutions_by_country.csv \\
        --topics-csv reports/aggregates/institutions_by_topic.csv \\
        --exclude-topics "linguistics,natural language processing,..." \\
        --top-countries 9 --top-institutions 4 --top-topics 3 \\
        --out reports/treemaps/treemap_3level_all_years.pdf \\
        --title "ACL Anthology — country → institution → topic"
"""
from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("treemap_3level")


def _luminance(r: float, g: float, b: float) -> float:
    def f(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _text_color_for_bg(rgba) -> str:
    return "white" if _luminance(*rgba[:3]) < 0.5 else "black"


def _wrap(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    lines = textwrap.wrap(text, width=max_chars,
                          break_long_words=False,
                          break_on_hyphens=False)
    return "\n".join(lines) if lines else text


def _make_palette(n: int):
    import matplotlib.cm as cm
    bases = [cm.tab20, cm.tab20b, cm.tab20c]
    colors = []
    for cmap in bases:
        for i in range(cmap.N):
            colors.append(cmap(i))
            if len(colors) >= n:
                return colors
    return colors


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insts-csv", required=True, type=Path,
                    help="institutions_by_country.csv")
    ap.add_argument("--topics-csv", required=True, type=Path,
                    help="institutions_by_topic.csv")
    ap.add_argument("--top-countries", type=int, default=9)
    ap.add_argument("--top-institutions", type=int, default=8,
                    help="Top institutions per country. Default 8 covers "
                         "~2x more institutions than the conservative 4 "
                         "without crushing the topic sub-cells.")
    ap.add_argument("--top-topics", type=int, default=5,
                    help="Top topics per institution. Default 5 — the "
                         "displayed topics are sized within the "
                         "institution rect, and each cell's percentage "
                         "is reported as a share of the institution's "
                         "TOTAL topic-tags (so the displayed top-5 may "
                         "sum to e.g. 60%% — the rest is implicit tail). "
                         "Set to 0 to show ALL topics with no truncation.")
    ap.add_argument("--include-other", dest="include_other",
                    action="store_true", default=False,
                    help="If set, append a synthetic 'other' cell holding "
                         "the truncated tail's sum (off by default — the "
                         "topic %% already discloses the tail implicitly).")
    ap.add_argument("--min-topic-frac", type=float, default=0.0,
                    help="Drop topic sub-cells whose share of the "
                         "institution's topic-tags is below this fraction. "
                         "0 (default) = keep everything.")
    ap.add_argument("--exclude-topics", default="",
                    help="Comma-separated topic labels to drop (umbrella).")
    ap.add_argument("--country-strip-frac", type=float, default=0.075)
    ap.add_argument("--inst-strip-frac", type=float, default=0.20,
                    help="Fraction of each institution rect reserved for "
                         "its name strip. Tuned to leave most of the "
                         "rect for the topic sub-cells while keeping the "
                         "institution name + share readable.")
    ap.add_argument("--inst-gutter", type=float, default=0.35,
                    help="Width (in data units, canvas is 100×100) of the "
                         "white inset between adjacent institutions inside "
                         "the same country. Higher = more visual "
                         "separation between institutions.")
    ap.add_argument("--max-font-size", type=float, default=14)
    ap.add_argument("--min-font-size", type=float, default=5.5)
    ap.add_argument("--color-by", choices=["region", "tab20"], default="region",
                    help="Outer-country colour scheme. 'region' colours "
                         "countries by continent/region (default).")
    ap.add_argument("--figsize", default="20,12")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import pandas as pd
    import squarify
    from _text_fitter import fit_label

    df_insts = pd.read_csv(args.insts_csv)
    df_topics = pd.read_csv(args.topics_csv)

    excluded = {s.strip() for s in args.exclude_topics.split(",") if s.strip()}
    df_topics = df_topics[~df_topics["cso_label"].isin(excluded)]
    df_topics = df_topics[df_topics["cso_label"].notna()
                          & (df_topics["cso_label"].astype(str).str.len() > 0)]

    # ---- Build country → [institutions] picks ----
    df_insts = df_insts[df_insts["canonical_name"].notna()
                        & (df_insts["canonical_name"].astype(str).str.len() > 0)]
    df_insts = df_insts[df_insts["paper_count"].fillna(0) > 0]

    country_totals = (df_insts.groupby("country_label")["paper_count"]
                              .sum().sort_values(ascending=False))
    top_countries = country_totals.head(args.top_countries).index.tolist()

    # Ensure every continent represented by at least one country in
    # the data appears in the plot (top-1 of any missing continent).
    from _region_palette import region_for as _region_for
    represented = {_region_for(c) for c in top_countries}
    for c in country_totals.index:
        r = _region_for(c)
        if r == "Other":
            continue
        if r not in represented:
            top_countries.append(c)
            represented.add(r)
            logger.info("  added %s (top of %s) for continent coverage", c, r)
    logger.info("top countries: %s", ", ".join(map(str, top_countries)))

    # For each (country, institution) pair we have paper_count.
    # For each (institution, topic) pair we have topic-tag count.
    # ror_id is the join key; institutions_by_topic also has ror_id and
    # canonical_name + country_label.
    topic_by_ror = df_topics.set_index(["ror_id", "cso_label"])["paper_count"]

    # 3-tuple per entry: (country, country_full_total, inst_list).
    # country_full_total is the FULL paper count across all institutions
    # for that country, used for honest country sizing/labels + an
    # "other institutions" tail cell that fills the country rect.
    structure: List[Tuple[str, float, List[Tuple[str, float, List[Tuple[str, float]]]]]] = []

    for country in top_countries:
        c_all = df_insts[df_insts["country_label"] == country]
        country_full = float(c_all["paper_count"].sum())
        c_df = c_all.sort_values("paper_count", ascending=False) \
                    .head(args.top_institutions)
        inst_list = []
        for _, row in c_df.iterrows():
            ror = row["ror_id"]
            inst_name = str(row["canonical_name"])
            inst_papers = float(row["paper_count"])

            # Topics for this institution
            full_t_df = (df_topics[df_topics["ror_id"] == ror]
                         .sort_values("paper_count", ascending=False))
            total_topic_tags = float(full_t_df["paper_count"].sum())
            t_df = full_t_df.copy()
            if args.top_topics and args.top_topics > 0:
                t_df = t_df.head(args.top_topics)
            if args.min_topic_frac > 0 and not t_df.empty:
                cutoff = total_topic_tags * args.min_topic_frac
                t_df = t_df[t_df["paper_count"] >= cutoff]
            topic_pairs = list(zip(t_df["cso_label"].astype(str),
                                   t_df["paper_count"].astype(float)))

            # Optional "+N other topics" tail (off by default)
            shown_sum = sum(v for _, v in topic_pairs)
            tail_sum = total_topic_tags - shown_sum
            tail_count = len(full_t_df) - len(topic_pairs)
            if (args.include_other and tail_sum > 0 and tail_count > 0):
                other_label = (f"+{tail_count} other topics"
                               if tail_count > 1
                               else "+1 other topic")
                topic_pairs.append((other_label, float(tail_sum)))
            inst_list.append((inst_name, inst_papers, topic_pairs))

        # Country total for sizing = sum of displayed top-K only, so
        # the country rect fills with institution cells (no empty
        # space, no synthetic +other cell).
        country_top_k = sum(p for _, p, _ in inst_list)
        structure.append((country, country_top_k, inst_list))

    canvas_w, canvas_h = 100.0, 100.0
    country_labels_in_order = [c for c, _, _ in structure]
    if args.color_by == "region":
        from _region_palette import (
            country_colors_by_region, region_legend_entries)
        palette = country_colors_by_region(country_labels_in_order)
        legend_entries = region_legend_entries(country_labels_in_order)
    else:
        palette = _make_palette(len(structure))
        legend_entries = []

    fig_w, fig_h = (float(s) for s in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.invert_yaxis()
    ax.set_axis_off()

    # ---- Layer 1: outer squarify on countries ----
    # Outer sizing uses each country's TOP-K sum so the canvas fills
    # without empty space; the %s reported are share-of-displayed-top-K.
    outer_vals = [country_top_k or 1.0
                  for _, country_top_k, _ in structure]
    displayed_country_total = sum(outer_vals) or 1.0
    outer_sizes = squarify.normalize_sizes(outer_vals, canvas_w, canvas_h)
    outer_rects = squarify.squarify(outer_sizes, 0, 0, canvas_w, canvas_h)

    for (country, country_top_k, insts), country_total, rect, base_color in zip(
            structure, outer_vals, outer_rects, palette):
        x, y, dx, dy = rect["x"], rect["y"], rect["dx"], rect["dy"]
        country_pct = 100.0 * country_total / displayed_country_total

        c_strip_h = max(2.4, dy * args.country_strip_frac)
        c_strip_h = min(c_strip_h, dy * 0.4)
        c_body_y, c_body_dy = y + c_strip_h, dy - c_strip_h

        # Country title strip
        ax.add_patch(patches.Rectangle(
            (x, y), dx, c_strip_h,
            facecolor=base_color, edgecolor="white",
            linewidth=2.0, alpha=0.98))
        ax.add_patch(patches.Rectangle(
            (x, c_body_y), dx, c_body_dy,
            facecolor=base_color, edgecolor="white",
            linewidth=2.0, alpha=0.15))

        ct_fs = min(max(10, dx * 0.18, c_strip_h * 0.55), 15)
        ct_tc = _text_color_for_bg(base_color)
        ax.text(x + dx / 2, y + c_strip_h / 2,
                f"{country}  —  {country_pct:.1f}%",
                ha="center", va="center",
                fontsize=ct_fs, fontweight="bold", color=ct_tc,
                clip_on=True)

        if not insts:
            continue

        # ---- Layer 2: institutions inside country body ----
        # Top-K institutions fill the country body. %s reported are
        # "% of displayed top-K" (sums to 100% within the country).
        i_vals = [p for _, p, _ in insts]
        i_total = sum(i_vals) or 1.0
        i_sizes = squarify.normalize_sizes(i_vals, dx, c_body_dy)
        i_rects = squarify.squarify(
            i_sizes, x, c_body_y, dx, c_body_dy)

        # Darken the country base colour for the institution borders so
        # each institution reads as one bounded unit, clearly separate
        # from its neighbours.
        dark_border = (max(0, base_color[0] * 0.55),
                       max(0, base_color[1] * 0.55),
                       max(0, base_color[2] * 0.55), 0.95)

        for i_idx, ((inst_name, inst_papers, topic_pairs), i_rect) in \
                enumerate(zip(insts, i_rects)):
            ix, iy, idx_, idy = (i_rect["x"], i_rect["y"],
                                 i_rect["dx"], i_rect["dy"])

            # Inset (gutter) so adjacent institutions don't visually merge.
            # Half-gutter on each side; clamp so we never shrink a rect to
            # negative dimensions.
            g = min(args.inst_gutter, idx_ * 0.18, idy * 0.18)
            ix += g / 2; iy += g / 2
            idx_ -= g; idy -= g
            if idx_ <= 0 or idy <= 0:
                continue

            # Slightly varied shade per institution rank
            alpha = 0.80 - 0.30 * (i_idx / max(len(insts) - 1, 1))
            inst_body_color = (base_color[0], base_color[1],
                               base_color[2], max(0.30, alpha - 0.35))

            # Layout: name strip on top, topic body below
            i_strip_h = max(1.8, idy * args.inst_strip_frac)
            i_strip_h = min(i_strip_h, idy * 0.55)
            i_body_y, i_body_dy = iy + i_strip_h, idy - i_strip_h

            # Pale body fill (so topic sub-cells stand out on top of it)
            ax.add_patch(patches.Rectangle(
                (ix, iy), idx_, idy,
                facecolor=inst_body_color, edgecolor="none"))

            # Name strip — solid, saturated colour of the country palette,
            # taking the full top width of the institution rect. This is
            # the visual "header" that owns the topic cells below it.
            strip_color = (base_color[0], base_color[1], base_color[2],
                           min(0.98, alpha + 0.18))
            ax.add_patch(patches.Rectangle(
                (ix, iy), idx_, i_strip_h,
                facecolor=strip_color, edgecolor="none"))

            # Institution outer border (dark, thick) — makes the whole
            # institution rect (header + topic body) read as ONE unit.
            ax.add_patch(patches.Rectangle(
                (ix, iy), idx_, idy,
                facecolor="none", edgecolor=dark_border,
                linewidth=1.8, joinstyle="miter"))

            # A thin horizontal divider between the name strip and the
            # topic body, in the same dark border colour — so the eye
            # reads "header above / details below" rather than two
            # disconnected rectangles.
            ax.plot([ix, ix + idx_], [i_body_y, i_body_y],
                    color=dark_border, linewidth=0.9, solid_capstyle="butt")

            # Institution name + paper share — biggest font that fits
            # the strip rect, leaving room for the percentage line.
            inst_pct = 100.0 * inst_papers / i_total
            name_fs, wname = fit_label(
                inst_name, idx_, i_strip_h,
                min_fs=args.min_font_size, max_fs=args.max_font_size,
                extra_lines=1)
            label_text = f"{wname}\n{inst_pct:.1f}% of {country}"
            ax.text(ix + idx_ / 2, iy + i_strip_h / 2, label_text,
                    ha="center", va="center",
                    fontsize=name_fs, fontweight="bold",
                    color=_text_color_for_bg(strip_color),
                    linespacing=1.0, clip_on=True)

            if not topic_pairs:
                continue

            # ---- Layer 3: topics inside institution body ----
            # Inset the topic area slightly so the topic cells visibly
            # sit INSIDE the institution rather than touching its border.
            inner_pad = min(0.25, idx_ * 0.04, i_body_dy * 0.05)
            tb_x = ix + inner_pad
            tb_y = i_body_y + inner_pad
            tb_dx = max(0.1, idx_ - 2 * inner_pad)
            tb_dy = max(0.1, i_body_dy - 2 * inner_pad)

            t_vals = [v for _, v in topic_pairs]
            # Denominator for % is the institution's FULL topic-tag total
            # (computed above as `total_topic_tags`), not the sum of
            # displayed cells — so a top-5 cell labelled "language model
            # 18%" reads as "18% of all this institution's topic tags",
            # and the displayed top-5 may sum to less than 100%, which
            # honestly discloses the implicit tail.
            t_total = total_topic_tags or 1.0
            t_sizes = squarify.normalize_sizes(t_vals, tb_dx, tb_dy)
            t_rects = squarify.squarify(t_sizes, tb_x, tb_y, tb_dx, tb_dy)

            # Smallest cell that can still fit even a 3pt one-char label
            # in data-coords; cells smaller than this just get a colour
            # but no text so the long tail doesn't render as garbage.
            MIN_LABEL_W = 1.4
            MIN_LABEL_H = 0.9

            for t_idx, ((t_name, t_v), t_rect) in enumerate(
                    zip(topic_pairs, t_rects)):
                tx, ty, tdx, tdy = (t_rect["x"], t_rect["y"],
                                    t_rect["dx"], t_rect["dy"])
                if tdx <= 0 or tdy <= 0:
                    continue

                # Topic cell — opaque, slightly varying alpha so the
                # bigger cells stand out against the tail
                t_alpha = max(0.50, alpha - 0.10
                              - 0.30 * (t_idx / max(len(topic_pairs) - 1, 1)))
                tcolor = (base_color[0], base_color[1], base_color[2],
                          t_alpha)
                ax.add_patch(patches.Rectangle(
                    (tx, ty), tdx, tdy,
                    facecolor=tcolor, edgecolor="white",
                    linewidth=0.35 if tdx < 1.5 else 0.6))

                # Skip labels on cells too small to host even a 3pt char
                if tdx < MIN_LABEL_W or tdy < MIN_LABEL_H:
                    continue

                t_pct = 100.0 * t_v / t_total
                # Best-fit font + wrap, leaving 1 line for the % below
                t_fs, wt = fit_label(
                    t_name, tdx, tdy,
                    min_fs=args.min_font_size,
                    max_fs=args.max_font_size - 1,
                    extra_lines=1)
                label = f"{wt}\n{t_pct:.0f}%"
                ax.text(tx + tdx / 2, ty + tdy / 2,
                        label,
                        ha="center", va="center",
                        fontsize=t_fs,
                        color=_text_color_for_bg(tcolor),
                        linespacing=1.0, clip_on=True)

    if args.title:
        fig.suptitle(args.title, fontsize=17, fontweight="bold", y=0.995)

    if legend_entries and len(legend_entries) > 1:
        handles = [patches.Patch(facecolor=color, edgecolor="white",
                                 label=label)
                   for label, color in legend_entries]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 6), frameon=False,
                   bbox_to_anchor=(0.5, -0.02), fontsize=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
