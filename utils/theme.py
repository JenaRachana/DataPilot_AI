"""
Shared dark-theme palette for DataPilot AI.

Single source of truth for the Python side (Streamlit CSS injection in
``ui/app.py``, matplotlib/seaborn rcParams in ``utils/code_exec.py``). Keep in
sync with ``.streamlit/config.toml`` (``theme.*`` / ``theme.dark.*``), which
duplicates these hex values since TOML can't import Python.

Validated categorical/sequential/status palette — dark-mode CVD separation and
contrast checked against the dark surface below.
"""

SURFACE = "#1a1a19"       # card / chart surface
PAGE = "#0d0d0d"          # page background
PRIMARY = "#3987e5"       # accent (categorical slot 1 — blue)
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
TEXT_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"
BORDER = "rgba(255, 255, 255, 0.10)"

# Fixed hue order — never cycled/reordered per series.
CATEGORICAL = [
    "#3987e5",  # blue
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
    "#d55181",  # magenta
    "#d95926",  # orange
]

# One hue (blue), light -> dark, for continuous magnitude (heatmaps, etc.).
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
