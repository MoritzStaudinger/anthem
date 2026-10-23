"""4-level nested squarified treemap: Region → Country → Institution → Topic.

Adds a REGION layer on top of plot_treemap_3level.py. Each region gets
a distinct hue (see scripts/_region_palette.py); countries inside a
region are progressively lighter shades of that hue; institutions and
topic sub-cells inherit shading from the country.

Each level carries its own title strip with the level's percentage:
  - Region title:      "Europe — 27.4%"   (% of displayed-corpus papers)
  - Country title:     "Germany — 27.3%"  (% of region's papers)
  - Institution strip: "Saarland Univ.\\n8.4% of Germany"
  - Topic sub-cell:    "language model\\n22%"  (% of institution's
                        total topic-tags)

Defaults are tuned to keep things legible: only the regions that
actually appear in the top-N countries are drawn, with up to 4
institutions per country and 4 topics per institution. Bump the
knobs for a poster figure.

Usage:
    python scripts/plot_treemap_4level.py \\
        --insts-csv  reports/aggregates/institutions_by_country.csv \\
        --topics-csv reports/aggregates/institutions_by_topic.csv \\
        --exclude-topics "linguistics,natural language processing,..." \\
        --top-countries 15 --top-institutions 4 --top-topics 4 \\
        --out reports/treemaps/treemap_4level_all_years.pdf \\
        --title "ACL Anthology — region → country → institution → topic"
"""
from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("treemap_4level")


# Shared helpers reused from plot_treemap_3level for consistency
def _luminance(r: float, g: float, b: float) -> float:
    def f(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _text_color_for_bg(rgba) -> str:
    return "white" if _luminance(*rgba[:3]) < 0.5 else "black"


def _shade(base_rgb: Tuple[float, float, float],
           rank: int, total: int) -> Tuple[float, float, float, float]:
    """Lighten the base toward white as rank → total-1."""
    n = max(total - 1, 1)
    blend = (rank / n) * 0.45
    r, g, b = base_rgb
    return (r + (1 - r) * blend, g + (1 - g) * blend,
            b + (1 - b) * blend, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insts-csv", required=True, type=Path)
    ap.add_argument("--topics-csv", required=True, type=Path)
    ap.add_argument("--top-countries", type=int, default=20,
                    help="Top countries to consider (across all "
                         "continents). Any continent with at least one "
                         "country in the corpus gets at least its top-1 "
                         "country included even if it falls outside the "
                         "top-N — so Africa, South America, Oceania are "
                         "guaranteed visible if they appear at all.")
    ap.add_argument("--top-institutions", type=int, default=8,
                    help="MAX institutions per country. The actual "
                         "number rendered may be lower if the country "
                         "rect is too small (see --min-area-per-inst).")
    ap.add_argument("--min-area-per-inst", type=float, default=12.0,
                    help="Minimum area (data units², canvas is 100×100) "
                         "each institution rect needs. The renderer "
                         "auto-truncates institutions per country so "
                         "country_area / N >= this value — tiny "
                         "countries show only their top-1 or top-2, "
                         "big countries show up to --top-institutions. "
                         "Default 12.0 keeps every visible institution "
                         "comfortably labelable. Lower to pack more in.")
    ap.add_argument("--top-topics", type=int, default=3,
                    help="Top topics per institution. Set to 0 for all.")
    ap.add_argument("--exclude-topics", default="")
    ap.add_argument("--skip-continents-detail",
                    default="",
                    help="Comma-separated continent names that get a "
                         "title-only rendering (no country/institution/"
                         "topic breakdown). Default: empty — small "
                         "continents are auto-collapsed via "
                         "--min-continent-pct-for-detail.")
    ap.add_argument("--min-continent-pct-for-detail", type=float,
                    default=2.0,
                    help="Continents whose share of the displayed total "
                         "is below this percentage get a title-only "
                         "rendering (no country/institution/topic "
                         "breakdown), because their cells would be too "
                         "small to host any legible label. Default 2.0%%. "
                         "Set to 0 to force detail for every continent.")
    ap.add_argument("--name-only-institutions", default="",
                    help="Comma-separated institution names that should "
                         "be rendered as NAME ONLY (no topic sub-cells), "
                         "using the whole institution rect for the name. "
                         "Match is case-insensitive substring against "
                         "canonical_name from the CSV. Useful for "
                         "highlighting specific universities by name.")
    ap.add_argument("--min-inst-area-for-topics", type=float, default=6.0,
                    help="Institutions whose rect area (in data units²; "
                         "canvas is 100×100) is BELOW this threshold are "
                         "rendered as name-only — the whole institution "
                         "rect is used for the university name + share, "
                         "no topic sub-cells. Default 6.0. Set to 0 to "
                         "always show topics regardless of size.")
    ap.add_argument("--region-strip-frac", type=float, default=0.06)
    ap.add_argument("--country-strip-frac", type=float, default=0.075)
    ap.add_argument("--inst-strip-frac", type=float, default=0.22)
    ap.add_argument("--inst-gutter", type=float, default=0.25)
    ap.add_argument("--max-font-size", type=float, default=14)
    ap.add_argument("--min-font-size", type=float, default=6.0)
    ap.add_argument("--figsize", default="22,14")
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
    from matplotlib.transforms import Bbox, TransformedBbox
    from _region_palette import REGION_BASE_RGB, region_for
    from _text_fitter import (
        fit_label as _fit_label_raw,
        fit_label_rot as _fit_label_rot_raw,
    )

    # Convert data units to points so fit_label's internal char-width
    # / line-height ratios (which are in font-size units = points) are
    # comparable to the cell dimensions we pass in.
    fig_w_in, fig_h_in = [float(s) for s in args.figsize.split(",")]
    PT_PER_UNIT_W = fig_w_in * 72.0 / 100.0   # canvas is 100×100
    PT_PER_UNIT_H = fig_h_in * 72.0 / 100.0

    def fit_label(text, cell_w, cell_h, **kw):
        return _fit_label_raw(
            text, cell_w * PT_PER_UNIT_W, cell_h * PT_PER_UNIT_H, **kw)

    def fit_label_rot(text, cell_w, cell_h, **kw):
        # Rotation disabled by user request — always horizontal. We
        # still keep the (fs, text, rotation) tuple signature so the
        # callers don't need to be touched.
        fs, txt = _fit_label_raw(
            text, cell_w * PT_PER_UNIT_W, cell_h * PT_PER_UNIT_H, **kw)
        return (fs, txt, 0)

    def _clip_text(ax_, cx_, cy_, label, rect_xywh, **kw):
        """Render text and clip it to the data-coord rect bounds. The
        default `clip_on=True` only clips to the axes bbox — we want
        text to be clipped to its OWN cell so it can't overflow into
        adjacent cells (which is the only way the chart can have
        overlapping text)."""
        bx, by, bw, bh = rect_xywh
        clip_box = TransformedBbox(
            Bbox.from_extents(bx, by, bx + bw, by + bh),
            ax_.transData)
        t = ax_.text(cx_, cy_, label, **kw)
        t.set_clip_box(clip_box)
        t.set_clip_on(True)
        return t

    df_insts = pd.read_csv(args.insts_csv)
    df_topics = pd.read_csv(args.topics_csv)

    excluded = {s.strip() for s in args.exclude_topics.split(",") if s.strip()}
    skip_detail_set = {s.strip() for s in args.skip_continents_detail.split(",")
                       if s.strip()}
    name_only_patterns = [s.strip().lower()
                          for s in args.name_only_institutions.split(",")
                          if s.strip()]

    def _is_name_only(inst_name: str, rect_w: float, rect_h: float) -> bool:
        """Decide if this institution should be rendered with the name
        spanning the whole rect (no topic subdivision):
          - explicit match against --name-only-institutions, OR
          - rect area < --min-inst-area-for-topics.
        """
        lname = inst_name.lower()
        for pat in name_only_patterns:
            if pat in lname:
                return True
        if args.min_inst_area_for_topics > 0:
            if rect_w * rect_h < args.min_inst_area_for_topics:
                return True
        return False
    df_topics = df_topics[~df_topics["cso_label"].isin(excluded)]
    df_topics = df_topics[df_topics["cso_label"].notna()
                          & (df_topics["cso_label"].astype(str).str.len() > 0)]
    df_insts = df_insts[df_insts["canonical_name"].notna()
                        & (df_insts["canonical_name"].astype(str).str.len() > 0)]
    df_insts = df_insts[df_insts["paper_count"].fillna(0) > 0]

    # Pick top-N countries by total papers
    country_totals = (df_insts.groupby("country_label")["paper_count"]
                              .sum().sort_values(ascending=False))
    top_countries = country_totals.head(args.top_countries).index.tolist()

    # Ensure every continent that has at least one country in the
    # data is represented — add the top-1 country from each missing
    # continent so Africa, South America, Oceania always appear if
    # they have any papers.
    represented = {region_for(c) for c in top_countries}
    all_in_data = country_totals.index.tolist()
    for c in all_in_data:
        r = region_for(c)
        if r == "Other":
            continue
        if r not in represented:
            top_countries.append(c)
            represented.add(r)
            logger.info("  added %s (top of %s, %d papers) to ensure "
                        "continent represented", c, r, int(country_totals[c]))
    logger.info("top countries (%d, regions=%s): %s",
                len(top_countries), sorted(represented),
                ", ".join(map(str, top_countries)))

    # Group selected countries by region, preserve order = first appearance
    by_region: "OrderedDict[str, List[str]]" = OrderedDict()
    for c in top_countries:
        r = region_for(c)
        by_region.setdefault(r, []).append(c)
    logger.info("regions present: %s", ", ".join(by_region.keys()))

    # Build the per-country institution + topic structure.
    # `country_full_total` (the SUM across ALL institutions for that
    # country) is what we use for the outer squarify and for the
    # country/region %. The tail (countries' papers beyond the top-K
    # institutions) becomes a synthetic "+N other instit." cell so the
    # country rect stays honestly proportional.
    structure: "OrderedDict[str, List[Tuple[str, float, List[Tuple[str, float, float, List[Tuple[str, float]]]]]]]" = OrderedDict()
    # structure[region] = [
    #   (country, country_full_total,
    #     [(inst_name, inst_papers, inst_total_topic_tags, topic_pairs)])
    # ]

    # Tracks FULL totals (across all institutions in the data) so the
    # rendered % labels can be the honest absolute share — e.g.
    # "Saarland 6.9% of Germany", not "27.8% of the displayed top-4".
    country_full_totals: "OrderedDict[str, float]" = OrderedDict()

    # Continent / corpus totals computed from the WHOLE dataset (not
    # just displayed countries) so the % numbers are invariant to the
    # --top-countries setting. The paper version and the big version
    # report the same percentages.
    all_country_totals = df_insts.groupby("country_label")["paper_count"].sum()
    all_region_totals: "OrderedDict[str, float]" = OrderedDict()
    for _country, _papers in all_country_totals.items():
        _r = region_for(_country)
        if _r == "Other":
            continue
        all_region_totals[_r] = all_region_totals.get(_r, 0.0) + float(_papers)
    grand_corpus_total = sum(all_region_totals.values()) or 1.0

    for region, countries in by_region.items():
        country_list = []
        for c in countries:
            c_all = df_insts[df_insts["country_label"] == c]
            country_full_total = float(c_all["paper_count"].sum())
            country_full_totals[c] = country_full_total
            c_df = c_all.sort_values("paper_count", ascending=False) \
                        .head(args.top_institutions)
            inst_list = []
            for _, row in c_df.iterrows():
                ror = row["ror_id"]
                inst_name = str(row["canonical_name"])
                inst_papers = float(row["paper_count"])
                full_t_df = (df_topics[df_topics["ror_id"] == ror]
                             .sort_values("paper_count", ascending=False))
                total_topic_tags = float(full_t_df["paper_count"].sum())
                t_df = full_t_df.copy()
                if args.top_topics and args.top_topics > 0:
                    t_df = t_df.head(args.top_topics)
                topic_pairs = list(zip(t_df["cso_label"].astype(str),
                                       t_df["paper_count"].astype(float)))
                inst_list.append((inst_name, inst_papers, total_topic_tags,
                                  topic_pairs))

            # Country "total" for SIZING = top-K sum (so the chart
            # fills its canvas). Full-country numbers are kept
            # separately for the percentage labels.
            country_top_k_sum = sum(p for _, p, _, _ in inst_list)
            country_list.append((c, country_top_k_sum, inst_list))
        structure[region] = country_list

    canvas_w, canvas_h = 100.0, 100.0

    # --- Layer 1: regions ---
    # SIZING uses top-K sums (so the chart fills the canvas).
    # PERCENTAGE LABELS use FULL country/region totals so they're the
    # honest absolute share (e.g. Saarland 6.9% of Germany, not its
    # 27.8% share of the displayed top-4).
    region_totals = OrderedDict(
        (r, sum(country_top_k for _, country_top_k, _ in cs))
        for r, cs in structure.items())
    grand_total = sum(region_totals.values()) or 1.0

    region_full_totals = OrderedDict(
        (r, sum(country_full_totals[c] for c, _, _ in cs))
        for r, cs in structure.items())
    grand_full_total = sum(region_full_totals.values()) or 1.0
    region_sizes = squarify.normalize_sizes(
        list(region_totals.values()), canvas_w, canvas_h)
    region_rects = squarify.squarify(region_sizes, 0, 0,
                                     canvas_w, canvas_h)

    fig_w, fig_h = (float(s) for s in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.invert_yaxis()
    ax.set_axis_off()

    # ---- Pre-pass: pick ONE uniform continent-title font size ----
    # We compute the largest font that fits the title in each
    # continent's rect (using whichever layout would actually be used —
    # title-only block or top strip), then take the min so every
    # continent label renders at the same size.
    continent_title_fs = float("inf")
    for (region, country_list), r_rect in zip(
            structure.items(), region_rects):
        rx_, ry_, rdx_, rdy_ = (r_rect["x"], r_rect["y"],
                                r_rect["dx"], r_rect["dy"])
        region_full_ = all_region_totals.get(region, 0.0)
        region_pct_ = 100.0 * region_full_ / grand_corpus_total
        is_too_small_ = region_pct_ < args.min_continent_pct_for_detail
        if region in skip_detail_set or is_too_small_:
            fs_, _ = fit_label(
                f"{region}\n{region_pct_:.1f}%",
                rdx_, rdy_, min_fs=6, max_fs=24)
        else:
            r_strip_h_ = max(2.5, rdy_ * args.region_strip_frac)
            r_strip_h_ = min(r_strip_h_, rdy_ * 0.18)
            fs_, _ = fit_label(
                f"{region}  —  {region_pct_:.1f}%",
                rdx_, r_strip_h_, min_fs=6, max_fs=20)
        if fs_ < continent_title_fs:
            continent_title_fs = fs_
    if continent_title_fs == float("inf") or continent_title_fs <= 0:
        continent_title_fs = 12.0

    for (region, country_list), region_total, r_rect in zip(
            structure.items(), region_totals.values(), region_rects):
        rx, ry, rdx, rdy = (r_rect["x"], r_rect["y"],
                            r_rect["dx"], r_rect["dy"])
        # % labels use FULL data-wide totals (invariant to --top-countries)
        region_full = all_region_totals.get(region, 0.0)
        region_pct = 100.0 * region_full / grand_corpus_total
        base_rgb = REGION_BASE_RGB.get(region, REGION_BASE_RGB["Other"])
        region_rgba = (*base_rgb, 1.0)

        # Region title strip
        r_strip_h = max(2.5, rdy * args.region_strip_frac)
        r_strip_h = min(r_strip_h, rdy * 0.18)
        r_body_y, r_body_dy = ry + r_strip_h, rdy - r_strip_h

        ax.add_patch(patches.Rectangle(
            (rx, ry), rdx, r_strip_h,
            facecolor=region_rgba, edgecolor="white", linewidth=2.5))
        ax.add_patch(patches.Rectangle(
            (rx, r_body_y), rdx, r_body_dy,
            facecolor=region_rgba, edgecolor="white", linewidth=2.5,
            alpha=0.08))

        # Collapse to title-only when:
        #  - the continent is explicitly listed in --skip-continents-detail, OR
        #  - its share of the displayed total is below
        #    --min-continent-pct-for-detail (auto rule).
        # Rationale: in a tiny continent panel every country/institution/
        # topic cell is below the legibility floor, so the panel ends
        # up visually empty even when the loop "renders" the inner
        # detail. A centred continent label uses the available space.
        is_too_small = region_pct < args.min_continent_pct_for_detail
        if region in skip_detail_set or is_too_small:
            _, big_text = fit_label(
                f"{region}\n{region_pct:.1f}%",
                rdx, rdy, min_fs=continent_title_fs,
                max_fs=continent_title_fs)
            _clip_text(ax, rx + rdx / 2, ry + rdy / 2,
                       big_text, (rx, ry, rdx, rdy),
                       ha="center", va="center",
                       fontsize=continent_title_fs, fontweight="bold",
                       color="black")
            continue

        _, rt_text = fit_label(
            f"{region}  —  {region_pct:.1f}%",
            rdx, r_strip_h,
            min_fs=continent_title_fs, max_fs=continent_title_fs)
        _clip_text(ax, rx + rdx / 2, ry + r_strip_h / 2,
                   rt_text, (rx, ry, rdx, r_strip_h),
                   ha="center", va="center",
                   fontsize=continent_title_fs, fontweight="bold",
                   color="black")

        # --- Layer 2: countries within the region ---
        country_totals_in_region = [
            (country_top_k or 1.0) for _, country_top_k, _ in country_list]
        country_sizes = squarify.normalize_sizes(
            country_totals_in_region, rdx, r_body_dy)
        country_rects = squarify.squarify(
            country_sizes, rx, r_body_y, rdx, r_body_dy)

        for ci, ((country, country_top_k, insts), c_total, c_rect) in \
                enumerate(zip(country_list, country_totals_in_region,
                              country_rects)):
            cx, cy, cdx, cdy = (c_rect["x"], c_rect["y"],
                                c_rect["dx"], c_rect["dy"])
            country_color = _shade(base_rgb, ci, len(country_list))
            # % of REAL continent total (data-wide), not displayed-share
            country_full = country_full_totals[country]
            country_pct_in_region = 100.0 * country_full / max(
                all_region_totals.get(region, 1.0), 1.0)

            c_strip_h = max(1.8, cdy * args.country_strip_frac)
            c_strip_h = min(c_strip_h, cdy * 0.20)
            c_body_y, c_body_dy = cy + c_strip_h, cdy - c_strip_h

            ax.add_patch(patches.Rectangle(
                (cx, cy), cdx, c_strip_h,
                facecolor=country_color, edgecolor="white", linewidth=1.5))
            ax.add_patch(patches.Rectangle(
                (cx, c_body_y), cdx, c_body_dy,
                facecolor=country_color, edgecolor="white",
                linewidth=1.5, alpha=0.18))

            # Country labels matter — keep them legible at the cost of
            # a small font on narrow country strips. clip_on guards
            # against any actual bleed-out.
            # Country labels cap at 85% of continent so they read as
            # the next level down, even when they wrap.
            ct_fs, ct_text, ct_rot = fit_label_rot(
                f"{country}  —  {country_pct_in_region:.1f}%",
                cdx, c_strip_h,
                min_fs=5.5, max_fs=continent_title_fs * 0.85)
            _clip_text(ax, cx + cdx / 2, cy + c_strip_h / 2,
                       ct_text, (cx, cy, cdx, c_strip_h),
                       ha="center", va="center",
                       fontsize=ct_fs, fontweight="bold",
                       rotation=ct_rot,
                       color="black")

            if not insts:
                continue

            # --- Layer 3: institutions within country ---
            # Auto-fit: only render as many institutions as can be
            # legibly drawn given the country's body area. Each
            # institution needs >= --min-area-per-inst data units².
            # Tiny countries thus show only their top-1 or top-2;
            # large countries show up to --top-institutions.
            country_body_area = cdx * c_body_dy
            if args.min_area_per_inst > 0:
                max_n_fitting = max(
                    1, int(country_body_area / args.min_area_per_inst))
                insts_to_show = insts[:max_n_fitting]
            else:
                insts_to_show = insts
            i_vals = [p for _, p, _, _ in insts_to_show]
            i_total = sum(i_vals) or 1.0
            i_sizes = squarify.normalize_sizes(i_vals, cdx, c_body_dy)
            i_rects = squarify.squarify(
                i_sizes, cx, c_body_y, cdx, c_body_dy)

            dark_border = (max(0, country_color[0] * 0.55),
                           max(0, country_color[1] * 0.55),
                           max(0, country_color[2] * 0.55), 0.95)

            for i_idx, ((inst_name, inst_papers, total_topic_tags,
                         topic_pairs), i_rect) in enumerate(
                    zip(insts_to_show, i_rects)):
                ix, iy, idx_, idy = (i_rect["x"], i_rect["y"],
                                     i_rect["dx"], i_rect["dy"])
                g = min(args.inst_gutter, idx_ * 0.15, idy * 0.15)
                ix += g / 2; iy += g / 2
                idx_ -= g; idy -= g
                if idx_ <= 0 or idy <= 0:
                    continue

                alpha = 0.80 - 0.25 * (i_idx / max(len(insts_to_show) - 1, 1))
                inst_body = (country_color[0], country_color[1],
                             country_color[2], max(0.30, alpha - 0.40))
                strip_c = (country_color[0], country_color[1],
                           country_color[2], min(0.95, alpha + 0.10))
                # % is share of FULL country, not displayed-top-K share
                inst_pct = 100.0 * inst_papers / max(country_full, 1)

                # --- Name-only mode: skip the strip layout and topic
                #     sub-cells, use the WHOLE institution rect for the
                #     institution name + percentage so the text gets
                #     the maximum possible font size. ---
                if _is_name_only(inst_name, idx_, idy):
                    name_only_color = (country_color[0], country_color[1],
                                       country_color[2],
                                       max(0.55, alpha))
                    ax.add_patch(patches.Rectangle(
                        (ix, iy), idx_, idy,
                        facecolor=name_only_color, edgecolor=dark_border,
                        linewidth=1.4))
                    fs, wt, rot = fit_label_rot(
                        inst_name, idx_, idy,
                        min_fs=args.min_font_size,
                        max_fs=continent_title_fs * 0.70,
                        extra_lines=1, skip_if_no_fit=True)
                    if fs > 0:
                        _clip_text(ax, ix + idx_ / 2, iy + idy / 2,
                                   f"{wt}\n{inst_pct:.1f}%",
                                   (ix, iy, idx_, idy),
                                   ha="center", va="center",
                                   fontsize=fs, fontweight="bold",
                                   rotation=rot, color="black",
                                   linespacing=1.0)
                    continue

                i_strip_h = max(1.4, idy * args.inst_strip_frac)
                i_strip_h = min(i_strip_h, idy * 0.6)
                i_body_y, i_body_dy = iy + i_strip_h, idy - i_strip_h

                # --- Pre-check the name fit, three-tier fallback:
                #   1. Name fits in strip → strip layout + topic cells
                #   2. Name doesn't fit in strip, but fits in WHOLE rect
                #      → render as name-only block (no topics)
                #   3. Name doesn't fit anywhere → solid coloured block
                # ---
                # Institution labels cap at 70% of continent
                name_fs, wname, name_rot = fit_label_rot(
                    inst_name, idx_, i_strip_h,
                    min_fs=args.min_font_size,
                    max_fs=continent_title_fs * 0.70,
                    extra_lines=1, skip_if_no_fit=True)

                if name_fs <= 0:
                    # Tier 2: try name in the whole rect (much bigger)
                    fs_full, wt_full, rot_full = fit_label_rot(
                        inst_name, idx_, idy,
                        min_fs=args.min_font_size,
                        max_fs=continent_title_fs * 0.70,
                        extra_lines=1, skip_if_no_fit=True)
                    name_only_color = (country_color[0], country_color[1],
                                       country_color[2],
                                       max(0.55, alpha))
                    ax.add_patch(patches.Rectangle(
                        (ix, iy), idx_, idy,
                        facecolor=name_only_color,
                        edgecolor=dark_border, linewidth=0.8))
                    if fs_full > 0:
                        _clip_text(ax, ix + idx_ / 2, iy + idy / 2,
                                   f"{wt_full}\n{inst_pct:.1f}%",
                                   (ix, iy, idx_, idy),
                                   ha="center", va="center",
                                   fontsize=fs_full, fontweight="bold",
                                   rotation=rot_full, color="black",
                                   linespacing=1.0)
                    continue

                ax.add_patch(patches.Rectangle(
                    (ix, iy), idx_, idy,
                    facecolor=inst_body, edgecolor="none"))
                ax.add_patch(patches.Rectangle(
                    (ix, iy), idx_, i_strip_h,
                    facecolor=strip_c, edgecolor="none"))
                ax.add_patch(patches.Rectangle(
                    (ix, iy), idx_, idy,
                    facecolor="none", edgecolor=dark_border,
                    linewidth=1.4, joinstyle="miter"))
                ax.plot([ix, ix + idx_], [i_body_y, i_body_y],
                        color=dark_border, linewidth=0.8)

                # name_fs / wname / name_rot were precomputed above
                # (we needed them before deciding whether to draw the
                # strip layout at all).
                label_text = f"{wname}\n{inst_pct:.1f}%"
                _clip_text(ax, ix + idx_ / 2, iy + i_strip_h / 2,
                           label_text, (ix, iy, idx_, i_strip_h),
                           ha="center", va="center",
                           fontsize=name_fs, fontweight="bold",
                           rotation=name_rot,
                           color="black",
                           linespacing=1.0)

                if not topic_pairs:
                    continue

                # --- Layer 4: topics within institution ---
                inner_pad = min(0.18, idx_ * 0.04, i_body_dy * 0.05)
                tb_x = ix + inner_pad
                tb_y = i_body_y + inner_pad
                tb_dx = max(0.1, idx_ - 2 * inner_pad)
                tb_dy = max(0.1, i_body_dy - 2 * inner_pad)

                t_vals = [v for _, v in topic_pairs]
                t_total = total_topic_tags or 1.0
                t_sizes = squarify.normalize_sizes(t_vals, tb_dx, tb_dy)
                t_rects = squarify.squarify(t_sizes, tb_x, tb_y, tb_dx, tb_dy)

                for t_idx, ((t_name, t_v), t_rect) in enumerate(
                        zip(topic_pairs, t_rects)):
                    tx, ty, tdx, tdy = (t_rect["x"], t_rect["y"],
                                        t_rect["dx"], t_rect["dy"])
                    if tdx <= 0 or tdy <= 0:
                        continue
                    t_alpha = max(0.55, alpha - 0.10
                                  - 0.20 * (t_idx
                                           / max(len(topic_pairs) - 1, 1)))
                    tcolor = (country_color[0], country_color[1],
                              country_color[2], t_alpha)
                    ax.add_patch(patches.Rectangle(
                        (tx, ty), tdx, tdy,
                        facecolor=tcolor, edgecolor="white",
                        linewidth=0.4))
                    # Only suppress labels on truly sub-pixel cells.
                    # Everything bigger gets at least a min-font label.
                    if tdx < 0.5 or tdy < 0.35:
                        continue

                    t_pct = 100.0 * t_v / t_total
                    # Skip topic labels that can't fit legibly — better
                    # to show a clean coloured cell than a clipped name.
                    # Topics cap at 55% of continent — smallest tier
                    t_fs, wt, t_rot = fit_label_rot(
                        t_name, tdx, tdy,
                        min_fs=args.min_font_size,
                        max_fs=continent_title_fs * 0.55,
                        extra_lines=1, skip_if_no_fit=True)
                    if t_fs <= 0:
                        continue
                    label = f"{wt}\n{t_pct:.0f}%"
                    _clip_text(ax, tx + tdx / 2, ty + tdy / 2,
                               label, (tx, ty, tdx, tdy),
                               ha="center", va="center",
                               fontsize=t_fs, rotation=t_rot,
                               color="black",
                               linespacing=1.0)

    if args.title:
        fig.suptitle(args.title, fontsize=17, fontweight="bold", y=0.995)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
