"""Country → continent → base-colour mapping shared by the treemap scripts.

Uses the standard 7-continent model: Africa, Antarctica, Asia, Europe,
North America, Oceania, South America. (Antarctica has no NLP papers,
so it never appears in practice.)

Countries in the same continent get the same hue; within a continent,
each country gets a slightly different lightness so adjacent panels
are still distinguishable. Anything outside the explicit map falls
back to a neutral grey labelled "Other".
"""
from __future__ import annotations

from typing import Dict, List, Tuple


# Country → continent. The 7-continent convention places transcontinental
# countries (Russia, Turkey, Israel) by where most of the population /
# academic activity lives or by long-standing convention:
#   - Russia → Europe (cultural / academic convention)
#   - Turkey → Asia   (geographically majority)
#   - Israel → Asia
COUNTRY_REGIONS: Dict[str, str] = {
    # --- North America ---
    "United States": "North America",
    "Canada":        "North America",
    "Mexico":        "North America",
    "Cuba":          "North America",
    "Dominican Republic": "North America",
    "Costa Rica":    "North America",
    "Panama":        "North America",
    "Guatemala":     "North America",

    # --- South America ---
    "Brazil":        "South America",
    "Argentina":     "South America",
    "Chile":         "South America",
    "Colombia":      "South America",
    "Peru":          "South America",
    "Venezuela":     "South America",
    "Uruguay":       "South America",
    "Ecuador":       "South America",
    "Bolivia":       "South America",
    "Paraguay":      "South America",

    # --- Europe ---
    "United Kingdom": "Europe",
    "Germany":        "Europe",
    "France":         "Europe",
    "Italy":          "Europe",
    "Netherlands":    "Europe",
    "Spain":          "Europe",
    "Switzerland":    "Europe",
    "Sweden":         "Europe",
    "Ireland":        "Europe",
    "Belgium":        "Europe",
    "Austria":        "Europe",
    "Denmark":        "Europe",
    "Finland":        "Europe",
    "Norway":         "Europe",
    "Poland":         "Europe",
    "Czechia":        "Europe",
    "Portugal":       "Europe",
    "Greece":         "Europe",
    "Hungary":        "Europe",
    "Romania":        "Europe",
    "Slovakia":       "Europe",
    "Slovenia":       "Europe",
    "Croatia":        "Europe",
    "Bulgaria":       "Europe",
    "Estonia":        "Europe",
    "Latvia":         "Europe",
    "Lithuania":      "Europe",
    "Serbia":         "Europe",
    "Iceland":        "Europe",
    "Luxembourg":     "Europe",
    "Malta":          "Europe",
    "Ukraine":        "Europe",
    "Belarus":        "Europe",
    "Russia":         "Europe",   # by convention
    "Bosnia and Herzegovina": "Europe",
    "Montenegro":     "Europe",
    "North Macedonia": "Europe",
    "Albania":        "Europe",
    "Cyprus":         "Europe",

    # --- Asia (unified: East / South / Southeast / Central / West / Middle East) ---
    "China":           "Asia",
    "Japan":           "Asia",
    "South Korea":     "Asia",
    "North Korea":     "Asia",
    "Hong Kong":       "Asia",
    "Taiwan":          "Asia",
    "Mongolia":        "Asia",
    "India":           "Asia",
    "Pakistan":        "Asia",
    "Bangladesh":      "Asia",
    "Sri Lanka":       "Asia",
    "Nepal":           "Asia",
    "Bhutan":          "Asia",
    "Maldives":        "Asia",
    "Afghanistan":     "Asia",
    "Singapore":       "Asia",
    "Malaysia":        "Asia",
    "Thailand":        "Asia",
    "Indonesia":       "Asia",
    "Vietnam":         "Asia",
    "Philippines":     "Asia",
    "Myanmar":         "Asia",
    "Cambodia":        "Asia",
    "Laos":            "Asia",
    "Brunei":          "Asia",
    "Israel":          "Asia",
    "Turkey":          "Asia",
    "Saudi Arabia":    "Asia",
    "United Arab Emirates": "Asia",
    "Qatar":           "Asia",
    "Bahrain":         "Asia",
    "Kuwait":          "Asia",
    "Oman":            "Asia",
    "Yemen":           "Asia",
    "Iran":            "Asia",
    "Iraq":            "Asia",
    "Jordan":          "Asia",
    "Lebanon":         "Asia",
    "Syria":           "Asia",
    "Palestine":       "Asia",
    "Kazakhstan":      "Asia",
    "Uzbekistan":      "Asia",
    "Turkmenistan":    "Asia",
    "Kyrgyzstan":      "Asia",
    "Tajikistan":      "Asia",
    "Armenia":         "Asia",
    "Azerbaijan":      "Asia",
    "Georgia":         "Asia",

    # --- Oceania ---
    "Australia":       "Oceania",
    "New Zealand":     "Oceania",
    "Fiji":            "Oceania",
    "Papua New Guinea": "Oceania",

    # --- Africa ---
    "South Africa":    "Africa",
    "Egypt":           "Africa",
    "Nigeria":         "Africa",
    "Kenya":           "Africa",
    "Morocco":         "Africa",
    "Tunisia":         "Africa",
    "Algeria":         "Africa",
    "Ethiopia":        "Africa",
    "Ghana":           "Africa",
    "Tanzania":        "Africa",
    "Uganda":          "Africa",
    "Rwanda":          "Africa",
    "Senegal":         "Africa",
    "Côte d'Ivoire":   "Africa",
    "Cameroon":        "Africa",
    "Sudan":           "Africa",
    "Libya":           "Africa",
    "Zambia":          "Africa",
    "Zimbabwe":        "Africa",
    "Botswana":        "Africa",
    "Mauritius":       "Africa",
    "Mozambique":      "Africa",
}


# One distinct base hue per continent. Inspired loosely by the Olympic
# colour scheme but tuned so all six populated continents stay
# distinguishable.
REGION_BASE_RGB: Dict[str, Tuple[float, float, float]] = {
    # Bright enough that BLACK text is legible on every base hue
    # (luminance > 0.5 for all six populated continents).
    "North America":  (0.93, 0.45, 0.45),   # warm red
    "South America":  (0.85, 0.58, 0.30),   # burnt orange / brown
    "Europe":         (0.52, 0.68, 0.92),   # bright steel blue
    "Asia":           (0.97, 0.65, 0.30),   # orange
    "Africa":         (0.50, 0.80, 0.55),   # green
    "Oceania":        (0.45, 0.80, 0.85),   # teal
    "Antarctica":     (0.78, 0.78, 0.88),   # pale violet (unused)
    "Other":          (0.78, 0.78, 0.78),   # neutral grey
}


# Fallback: if the aggregator didn't translate an ISO-2 code to a
# friendly name, map the code directly. Built from the same data set
# as COUNTRY_REGIONS above.
ISO2_REGIONS: Dict[str, str] = {
    # North America
    "US": "North America", "CA": "North America", "MX": "North America",
    "CU": "North America", "DO": "North America", "CR": "North America",
    "PA": "North America", "GT": "North America",
    # South America
    "BR": "South America", "AR": "South America", "CL": "South America",
    "CO": "South America", "PE": "South America", "VE": "South America",
    "UY": "South America", "EC": "South America", "BO": "South America",
    "PY": "South America",
    # Europe
    "GB": "Europe", "DE": "Europe", "FR": "Europe", "IT": "Europe",
    "NL": "Europe", "ES": "Europe", "CH": "Europe", "SE": "Europe",
    "IE": "Europe", "BE": "Europe", "AT": "Europe", "DK": "Europe",
    "FI": "Europe", "NO": "Europe", "PL": "Europe", "CZ": "Europe",
    "PT": "Europe", "GR": "Europe", "HU": "Europe", "RO": "Europe",
    "SK": "Europe", "SI": "Europe", "HR": "Europe", "BG": "Europe",
    "EE": "Europe", "LV": "Europe", "LT": "Europe", "RS": "Europe",
    "IS": "Europe", "LU": "Europe", "MT": "Europe", "UA": "Europe",
    "BY": "Europe", "RU": "Europe", "CY": "Europe",
    # Asia (unified)
    "CN": "Asia", "JP": "Asia", "KR": "Asia", "KP": "Asia",
    "HK": "Asia", "TW": "Asia", "MN": "Asia",
    "IN": "Asia", "PK": "Asia", "BD": "Asia", "LK": "Asia",
    "NP": "Asia", "BT": "Asia", "MV": "Asia", "AF": "Asia",
    "SG": "Asia", "MY": "Asia", "TH": "Asia", "ID": "Asia",
    "VN": "Asia", "PH": "Asia", "MM": "Asia", "KH": "Asia",
    "LA": "Asia", "BN": "Asia",
    "IL": "Asia", "TR": "Asia", "SA": "Asia", "AE": "Asia",
    "QA": "Asia", "BH": "Asia", "KW": "Asia", "OM": "Asia",
    "YE": "Asia", "IR": "Asia", "IQ": "Asia", "JO": "Asia",
    "LB": "Asia", "SY": "Asia", "PS": "Asia",
    "KZ": "Asia", "UZ": "Asia", "TM": "Asia", "KG": "Asia",
    "TJ": "Asia", "AM": "Asia", "AZ": "Asia", "GE": "Asia",
    # Oceania
    "AU": "Oceania", "NZ": "Oceania", "FJ": "Oceania", "PG": "Oceania",
    # Africa
    "ZA": "Africa", "EG": "Africa", "NG": "Africa", "KE": "Africa",
    "MA": "Africa", "TN": "Africa", "DZ": "Africa", "ET": "Africa",
    "GH": "Africa", "TZ": "Africa", "UG": "Africa", "RW": "Africa",
    "SN": "Africa", "CI": "Africa", "CM": "Africa", "SD": "Africa",
    "LY": "Africa", "ZM": "Africa", "ZW": "Africa", "BW": "Africa",
    "MU": "Africa", "MZ": "Africa",
}


def region_for(country_label: str) -> str:
    if country_label in COUNTRY_REGIONS:
        return COUNTRY_REGIONS[country_label]
    if (isinstance(country_label, str)
            and len(country_label) == 2 and country_label.isupper()):
        return ISO2_REGIONS.get(country_label, "Other")
    return "Other"


def country_colors_by_region(country_labels: List[str]
                             ) -> List[Tuple[float, float, float, float]]:
    """Return one RGBA per country. Countries in the same continent
    share a *family* of colours (similar hue) but are made visibly
    distinct by varying HSV hue + saturation + value across the
    countries within that continent.

    Within a continent the strongest variation is *hue* (±35° around
    the base hue) so that adjacent countries like Taiwan / Singapore /
    Israel / UAE in Asia don't all look like "another shade of
    orange". Lightness varies as a secondary signal.
    """
    import colorsys

    region_to_indices: Dict[str, List[int]] = {}
    region_order: List[str] = []
    for i, c in enumerate(country_labels):
        r = region_for(c)
        if r not in region_to_indices:
            region_to_indices[r] = []
            region_order.append(r)
        region_to_indices[r].append(i)

    out: List[Tuple[float, float, float, float]] = [None] * len(country_labels)
    for r in region_order:
        idxs = region_to_indices[r]
        base = REGION_BASE_RGB.get(r, REGION_BASE_RGB["Other"])
        base_h, base_s, base_v = colorsys.rgb_to_hsv(*base)
        n = max(len(idxs) - 1, 1)
        phi = 0.6180339887
        for rank, ci in enumerate(idxs):
            t = rank / n if len(idxs) > 1 else 0.0
            # Leader keeps the canonical continent hue so the region
            # is instantly recognisable. Subsequent countries are
            # spread via the golden ratio across a ±0.10 band so even
            # the 10th country in a continent is visibly distinct.
            if rank == 0:
                hue_step = 0.0
            else:
                hue_step = (((rank - 1) * phi) % 1.0) * 0.20 - 0.10
            new_h = (base_h + hue_step) % 1.0
            # Saturation: full for leader, gentle drop to a still-vivid floor
            new_s = max(0.65, base_s - 0.15 * t)
            # Value: leader at base, slight darkening for tail (keeps
            # the per-rank distinction without washing colours out)
            new_v = max(0.65, min(1.0, base_v - 0.20 * t))
            r_, g_, b_ = colorsys.hsv_to_rgb(new_h, new_s, new_v)
            out[ci] = (r_, g_, b_, 1.0)
    return out


def region_legend_entries(country_labels: List[str]
                          ) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """For each continent that appears among the displayed countries,
    return one (continent_name, base_color) entry — feed to a matplotlib
    legend so readers can see which colour means which continent."""
    seen: List[Tuple[str, Tuple[float, float, float, float]]] = []
    seen_regions = set()
    for c in country_labels:
        r = region_for(c)
        if r in seen_regions:
            continue
        seen_regions.add(r)
        base = REGION_BASE_RGB.get(r, REGION_BASE_RGB["Other"])
        seen.append((r, (base[0], base[1], base[2], 1.0)))
    return seen
