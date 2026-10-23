"""Render a grouped squarified treemap from any of the long-form CSVs
produced by scripts/aggregate_for_treemap.py.

The chart has two levels of nesting:

  - **Outer squarify** lays out one rectangle per *group* (e.g. country),
    sized proportional to the group's total paper count.
  - **Inner squarify** runs again inside each group rectangle to place
    one cell per *item* (e.g. institution), sized by that item's paper
    count within the group.

Readability features:

  - Per-cell font size scales with cell dimensions in data coordinates.
  - Text colour automatically flips between black and white based on
    the cell background's luminance, so labels stay legible whatever
    palette you pick.
  - Long names are wrapped to fit the cell width.
  - **Tiny cells are silently dropped from the label layer** if their
    width/height is below the legibility threshold *or* their share of
    the total area falls under `--min-label-area-frac` (default 0.5 %).
    The colour rectangle is still drawn, so the cell remains
    proportionally faithful — only the label disappears.

Dependencies (laptop side):
    pip install matplotlib pandas squarify

Examples:

    # Institutions × country, 2020-2025 slice — paper headline figure
    python scripts/plot_treemap.py \\
        --csv  reports/aggregates/2020_2025/institutions_by_country.csv \\
        --kind institutions_by_country \\
        --top-groups 12 --top-items-per-group 15 \\
        --out reports/treemaps/institutions_by_country_2020_2025.png \\
        --title "ACL Anthology 2020-2025 — top institutions per country"

    # Topics × country, exclude umbrella labels
    python scripts/plot_treemap.py \\
        --csv  reports/aggregates/2020_2025/topics_by_country.csv \\
        --kind topics_by_country \\
        --exclude "linguistics,natural language processing,natural languages,language model,computational linguistics,computer science,university,semantics" \\
        --top-groups 12 --top-items-per-group 12 \\
        --out reports/treemaps/topics_by_country_2020_2025.png

    # Institutions × topic
    python scripts/plot_treemap.py \\
        --csv  reports/aggregates/2020_2025/institutions_by_topic.csv \\
        --kind institutions_by_topic \\
        --top-groups 12 --top-items-per-group 15 \\
        --out reports/treemaps/institutions_by_topic_2020_2025.png
"""
from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("plot_treemap")


# Map --kind to (group_col, item_col, value_col, default title)
KINDS: Dict[str, Tuple[str, str, str, str]] = {
    "institutions_by_country": (
        "country_label", "canonical_name", "paper_count",
        "Institutions by country"),
    "topics_by_country": (
        "country_label", "cso_label", "paper_count",
        "Topics by country"),
    "institutions_by_topic": (
        "cso_label", "canonical_name", "paper_count",
        "Institutions by topic"),
}


def _luminance(r: float, g: float, b: float) -> float:
    """Relative luminance per WCAG, inputs in [0, 1]."""
    def f(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _text_color_for_bg(rgba) -> str:
    """Black or white, whichever has more contrast against rgba."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    return "white" if _luminance(r, g, b) < 0.5 else "black"


def _wrap_label(text: str, max_chars: int) -> str:
    """textwrap with a hard cap (no hyphenation) — keep words whole
    so 'Massachusetts Institute of Technology' breaks across lines
    rather than mid-word."""
    if max_chars <= 0:
        return text
    lines = textwrap.wrap(text, width=max_chars,
                          break_long_words=False,
                          break_on_hyphens=False)
    return "\n".join(lines) if lines else text


def _make_palette(n: int):
    """Return n visually distinct mpl colors. Combines tab20 + tab20b
    + tab20c so we have enough variety for ≤60 groups."""
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
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--kind", choices=list(KINDS.keys()),
                    help="Preset that sets group/item/value columns and "
                         "a default title.")
    ap.add_argument("--group-col", default=None,
                    help="Override the outer-grouping column "
                         "(otherwise comes from --kind).")
    ap.add_argument("--item-col", default=None)
    ap.add_argument("--value-col", default=None)
    ap.add_argument("--top-groups", type=int, default=12,
                    help="How many outer groups to render.")
    ap.add_argument("--top-items-per-group", type=int, default=15)
    ap.add_argument("--exclude", default="",
                    help="Comma-separated values to drop from the "
                         "ITEM column before ranking (e.g. drop "
                         "'University' as a fake institution).")
    ap.add_argument("--exclude-groups", default="",
                    help="Comma-separated values to drop from the "
                         "GROUP column before ranking — use this to "
                         "exclude umbrella CSO topics from "
                         "institutions_by_topic.")
    ap.add_argument("--show-percent", dest="show_percent",
                    action="store_true", default=True,
                    help="Append (X.X%%) to each cell label, computed "
                         "as item / displayed-group-total. On by default.")
    ap.add_argument("--no-percent", dest="show_percent",
                    action="store_false",
                    help="Disable per-cell percentage annotation.")
    ap.add_argument("--max-font-size", type=float, default=14)
    ap.add_argument("--min-font-size", type=float, default=5.0,
                    help="Floor for per-cell label font size. The script "
                         "labels EVERY cell — set this low if you want "
                         "the smallest tail cells to still get a "
                         "(possibly squint-readable) label.")
    ap.add_argument("--color-by", choices=["region", "tab20"], default=None,
                    help="Outer-group colour scheme. 'region' colours "
                         "countries by continent/region (only meaningful "
                         "when the outer group IS countries). Default: "
                         "'region' when --kind is institutions_by_country "
                         "or topics_by_country, else 'tab20'.")
    ap.add_argument("--title-strip-frac", type=float, default=0.08,
                    help="Fraction of each group rectangle's height "
                         "reserved for a non-overlapping title strip. "
                         "Default 0.08 = 8%%.")
    ap.add_argument("--figsize", default="16,10",
                    help="matplotlib figsize as 'W,H' in inches.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    # Heavy imports kept here so --help is snappy
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import pandas as pd
    import squarify
    from _text_fitter import fit_label

    # Resolve columns + default title from --kind if not overridden
    if args.kind:
        gc, ic, vc, default_title = KINDS[args.kind]
        group_col = args.group_col or gc
        item_col = args.item_col or ic
        value_col = args.value_col or vc
        title = args.title or default_title
    else:
        if not (args.group_col and args.item_col and args.value_col):
            raise SystemExit("either --kind or all of --group-col / "
                             "--item-col / --value-col must be given")
        group_col = args.group_col
        item_col = args.item_col
        value_col = args.value_col
        title = args.title or args.csv.stem.replace("_", " ")

    df = pd.read_csv(args.csv)
    for col in (group_col, item_col, value_col):
        if col not in df.columns:
            raise SystemExit(f"--csv missing column '{col}' "
                             f"(have: {list(df.columns)})")

    # Drop excluded items + excluded groups, blanks, zero-rows
    excluded_items = {s.strip() for s in args.exclude.split(",") if s.strip()}
    excluded_groups = {s.strip() for s in args.exclude_groups.split(",") if s.strip()}
    df = df[~df[item_col].isin(excluded_items)]
    df = df[~df[group_col].isin(excluded_groups)]
    df = df[df[item_col].notna() & (df[item_col].astype(str).str.len() > 0)]
    df = df[df[group_col].notna() & (df[group_col].astype(str).str.len() > 0)]
    df = df[df[value_col].fillna(0) > 0]

    # Outer ranking: top-N groups by FULL total (across all items in
    # the group, not just the displayed top-K).
    group_totals = (df.groupby(group_col)[value_col]
                      .sum().sort_values(ascending=False))
    top_groups = group_totals.head(args.top_groups).index.tolist()

    # When the outer grouping is countries, ensure every continent
    # with at least one country in the data appears (top-1 of any
    # missing continent). Skipped for topic-keyed plots.
    if args.kind in ("institutions_by_country", "topics_by_country"):
        from _region_palette import region_for as _region_for
        represented = {_region_for(c) for c in top_groups}
        for c in group_totals.index:
            r = _region_for(c)
            if r == "Other":
                continue
            if r not in represented:
                top_groups.append(c)
                represented.add(r)
                logger.info("  added %s (top of %s) for continent coverage",
                            c, r)

    logger.info("top %d groups: %s", len(top_groups),
                ", ".join(map(str, top_groups[:8])) +
                ("..." if len(top_groups) > 8 else ""))

    # Build per-group items (top-K within each) + an "+N other items"
    # tail cell so the inner squarify fills the group rect honestly
    # and item percentages are "% of group's full total".
    per_group: List[Tuple[str, List[Tuple[str, float]], float]] = []
    for g in top_groups:
        g_all = df[df[group_col] == g]
        full_group_total = float(group_totals[g])
        items = g_all.sort_values(value_col, ascending=False) \
                     .head(args.top_items_per_group)
        pairs = list(zip(items[item_col].astype(str),
                         items[value_col].astype(float)))
        shown_sum = sum(v for _, v in pairs)
        tail_sum = full_group_total - shown_sum
        tail_count = len(g_all) - len(pairs)
        if tail_sum > 0 and tail_count > 0:
            tail_label = (f"+{tail_count} other"
                          if tail_count == 1
                          else f"+{tail_count} others")
            pairs.append((tail_label, float(tail_sum)))
        per_group.append((str(g), pairs, full_group_total))

    canvas_w = 100.0
    canvas_h = 100.0

    # --- Outer squarify on groups ---
    outer_vals = [t[2] for t in per_group]
    displayed_total = sum(outer_vals) or 1.0
    outer_sizes = squarify.normalize_sizes(outer_vals, canvas_w, canvas_h)
    outer_rects = squarify.squarify(outer_sizes, 0, 0, canvas_w, canvas_h)

    # Default colour scheme: region-based for country-keyed plots,
    # tab20 fallback otherwise. Honours --color-by override.
    color_mode = args.color_by
    if color_mode is None:
        color_mode = ("region"
                      if args.kind in ("institutions_by_country",
                                       "topics_by_country")
                      else "tab20")
    if color_mode == "region":
        from _region_palette import (
            country_colors_by_region, region_legend_entries)
        country_labels = [g for g, _, _ in per_group]
        palette = country_colors_by_region(country_labels)
        legend_entries = region_legend_entries(country_labels)
    else:
        palette = _make_palette(len(per_group))
        legend_entries = []

    fig_w, fig_h = (float(s) for s in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.invert_yaxis()
    ax.set_axis_off()

    for (label, pairs, group_total), rect, color in zip(
            per_group, outer_rects, palette):
        x, y, dx, dy = rect["x"], rect["y"], rect["dx"], rect["dy"]
        group_pct = 100.0 * group_total / displayed_total

        # Reserve a strip at the top of the group rect for the title;
        # inner squarify happens in the area BELOW the strip so cell
        # labels never collide with the group title.
        strip_h = max(2.0, dy * args.title_strip_frac)
        strip_h = min(strip_h, dy * 0.5)  # safety: never eat >50% of group
        body_y, body_dy = y + strip_h, dy - strip_h

        # Title strip fill: same hue as the group, darker, with text on top
        ax.add_patch(patches.Rectangle(
            (x, y), dx, strip_h,
            facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.95))

        # Group body fill (very light) so adjacent groups read distinctly
        ax.add_patch(patches.Rectangle(
            (x, body_y), dx, body_dy,
            facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.18))

        # Group title text inside the strip — includes group's share of
        # the displayed top-N total so the first level reads as
        # proportional alongside the inner cells.
        strip_fs = min(max(9, dx * 0.18, strip_h * 0.55), 15)
        strip_tc = _text_color_for_bg(color)
        ax.text(x + dx / 2, y + strip_h / 2,
                f"{label}  —  {group_pct:.1f}%",
                ha="center", va="center",
                fontsize=strip_fs, fontweight="bold",
                color=strip_tc, clip_on=True)

        # --- Inner squarify on items of this group (in the BODY area) ---
        if not pairs:
            continue
        inner_vals = [v for _, v in pairs]
        inner_sizes = squarify.normalize_sizes(inner_vals, dx, body_dy)
        inner_rects = squarify.squarify(
            inner_sizes, x, body_y, dx, body_dy)

        # Per-panel total — used for the inner-cell %% so each panel's
        # percentages sum to 100. Distinct from the outer-loop's
        # `displayed_total`, which is the sum across all displayed
        # groups and feeds the group title.
        panel_total = sum(inner_vals) or 1.0

        n_items = len(pairs)
        for i, ((iname, ival), irect) in enumerate(zip(pairs, inner_rects)):
            ix, iy, idx, idy = (irect["x"], irect["y"],
                                irect["dx"], irect["dy"])
            # Shade rank: largest cell darkest, smallest cell lightest
            alpha = 0.85 - 0.45 * (i / max(n_items - 1, 1))
            cell_color = (color[0], color[1], color[2], alpha)
            ax.add_patch(patches.Rectangle(
                (ix, iy), idx, idy,
                facecolor=cell_color, edgecolor="white", linewidth=0.5))

            # Optimal text fit: pick the largest font size that lets
            # the wrapped name + an extra pct line fit inside the cell.
            pct_str = (f"{100.0 * ival / panel_total:.1f}%"
                       if args.show_percent else "")
            extra = 1 if pct_str else 0
            fs, wrapped_name = fit_label(
                iname, idx, idy,
                min_fs=args.min_font_size, max_fs=args.max_font_size,
                extra_lines=extra)
            text = (wrapped_name + ("\n" + pct_str if pct_str else ""))

            tc = _text_color_for_bg(cell_color)
            ax.text(ix + idx / 2, iy + idy / 2, text,
                    ha="center", va="center",
                    fontsize=fs, color=tc,
                    linespacing=1.0, clip_on=True)

    if title:
        fig.suptitle(title, fontsize=17, fontweight="bold", y=0.995)

    # Region legend at the bottom — only when --color-by region is in
    # effect and there's more than one region in the data.
    if legend_entries and len(legend_entries) > 1:
        handles = [patches.Patch(facecolor=color, edgecolor="white",
                                 label=label)
                   for label, color in legend_entries]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 6), frameon=False,
                   bbox_to_anchor=(0.5, -0.02), fontsize=11)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
