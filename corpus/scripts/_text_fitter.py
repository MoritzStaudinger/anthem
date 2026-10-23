"""Best-fit text sizing for treemap cell labels.

Given a label string and a cell's (width, height) in data coordinates,
pick the (font_size, wrapped_text) combination that fills as much of
the cell as possible while still fitting.

The naive `font_size = min(W*0.5, H*0.4)` heuristic is conservative —
big cells leave huge empty margins because the heuristic doesn't know
that text could wrap onto more lines and use the leftover height. This
fitter tries every plausible number of lines and picks whichever wrap
yields the largest legible font.

API:
    fit_label(text, cell_w, cell_h,
              min_fs=5.0, max_fs=14.0,
              w_ratio=0.92, h_ratio=0.85,
              extra_lines=0)
        → (font_size, wrapped_text)

    fit_label_rot(text, cell_w, cell_h, ...)
        → (font_size, wrapped_text, rotation)
        Same as fit_label but also tries 90° rotation and returns
        whichever orientation yields the larger font.
"""
from __future__ import annotations

import textwrap
from typing import Tuple

# Conservative font-metric constants (sans-serif, roughly matches
# matplotlib's DejaVu Sans Bold). Tuned higher than the textbook 0.55
# because bold sans-serif glyphs are visibly wider than the average,
# and the calculated font size needs a safety margin so the rendered
# text doesn't bleed past its cell on either end.
_CHAR_W_PER_PT = 0.70       # average glyph width in font-size units
_LINE_H_PER_PT = 1.18       # line height in font-size units


def fit_label(text: str,
              cell_w: float,
              cell_h: float,
              min_fs: float = 5.0,
              max_fs: float = 14.0,
              w_ratio: float = 0.95,
              h_ratio: float = 0.92,
              extra_lines: int = 0,
              skip_if_no_fit: bool = False) -> Tuple[float, str]:
    """Return (font_size, wrapped_text) that maximises font_size such
    that the wrapped text fits in (cell_w × cell_h) data units,
    clamped to [min_fs, max_fs].

    `extra_lines` reserves vertical space for additional lines you'll
    append yourself (e.g. a percentage line in a smaller font).

    `skip_if_no_fit` — when True, return (0.0, "") if the algorithm
    can't fit the text at or above min_fs. The caller can use this
    to skip drawing the label entirely on cells too small to host
    legible text (rather than getting an illegible min_fs render).
    """
    if not text or cell_w <= 0 or cell_h <= 0:
        return (min_fs, text or "")

    avail_w = max(cell_w * w_ratio, 1e-6)
    avail_h = max(cell_h * h_ratio, 1e-6)

    words = text.split()
    if not words:
        return (min_fs, text)
    longest_word = max(len(w) for w in words)

    best_fs = 0.0
    best_text = text

    # Try line counts from 1 to N where N is bounded by both the word
    # count and a sane upper limit so we don't pay for absurd wraps.
    max_lines_to_try = min(len(words), 6)
    for n_lines in range(1, max_lines_to_try + 1):
        # Target chars-per-line is roughly len(text) / n_lines, but it
        # can't be shorter than the longest word (textwrap won't break
        # words mid-token), and we need at least a few chars of headroom.
        target_chars = max(longest_word + 1,
                           (len(text) + n_lines - 1) // n_lines)

        wrapped = textwrap.wrap(text, width=target_chars,
                                break_long_words=False,
                                break_on_hyphens=False)
        if not wrapped:
            continue

        actual_lines = len(wrapped) + extra_lines
        actual_max_line = max(len(line) for line in wrapped)
        if actual_lines <= 0 or actual_max_line <= 0:
            continue

        fs_w = avail_w / (actual_max_line * _CHAR_W_PER_PT)
        fs_h = avail_h / (actual_lines * _LINE_H_PER_PT)
        fs = min(fs_w, fs_h)

        if fs > best_fs:
            best_fs = fs
            best_text = "\n".join(wrapped)

    if best_fs <= 0:
        if skip_if_no_fit:
            return (0.0, "")
        return (min_fs, text)
    if best_fs < min_fs and skip_if_no_fit:
        return (0.0, "")
    fs = max(min_fs, min(max_fs, best_fs))
    return (fs, best_text)


def fit_label_rot(text: str,
                  cell_w: float,
                  cell_h: float,
                  min_fs: float = 5.0,
                  max_fs: float = 14.0,
                  w_ratio: float = 0.95,
                  h_ratio: float = 0.92,
                  extra_lines: int = 0,
                  rotation_threshold: float = 1.15,
                  skip_if_no_fit: bool = False) -> Tuple[float, str, int]:
    """Like fit_label but also tries a 90° rotation and returns the
    orientation that yields the larger legible font.

    Returns (font_size, wrapped_text, rotation_degrees).
    Only rotates when the vertical fit is at least
    `rotation_threshold` × better than horizontal — keeps the default
    horizontal where there's no real win, since rotated text is
    harder to read at a glance.
    """
    fs_h, text_h = fit_label(text, cell_w, cell_h,
                             min_fs=min_fs, max_fs=max_fs,
                             w_ratio=w_ratio, h_ratio=h_ratio,
                             extra_lines=extra_lines,
                             skip_if_no_fit=skip_if_no_fit)
    # Vertical: swap width and height
    fs_v, text_v = fit_label(text, cell_h, cell_w,
                             min_fs=min_fs, max_fs=max_fs,
                             w_ratio=w_ratio, h_ratio=h_ratio,
                             extra_lines=extra_lines,
                             skip_if_no_fit=skip_if_no_fit)
    if fs_v > fs_h * rotation_threshold:
        return (fs_v, text_v, 90)
    return (fs_h, text_h, 0)

