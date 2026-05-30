"""Unified plot style for all figure scripts.

Single source of truth for fonts, colours, and figure-saving conventions
so all figures in the thesis share one visual language.

Usage:
    from plot_style import apply_style, MODEL_COLORS, METHOD_COLORS, savefig_both

    apply_style()
    ...
    savefig_both(os.path.join(OUTPUT_DIR, "fig_01_rank_frequency.png"))
"""

import matplotlib.pyplot as plt


PAPER_SLOPE = -1.237        # Michaud et al. (2023) target
EXPECTED_SLOPE = -1.083     # n=10000 token-budget correction


# Models: blue/red for the small-to-large size axis; Methods use a distinct palette

MODEL_COLORS = {
    "Pythia-19m":  "#1f77b4",
    "Pythia-125m": "#d62728",
}

# Wong colorblind-safe pair; distinct from MODEL_COLORS to avoid double-encoding
HUMAN_AI_COLORS = {
    "Human": "#009E73",          # teal-green
    "AI":    "#D55E00",          # vermillion
}

METHOD_COLORS = {
    "Spectral":     "#555555",   # baseline (paper's method) - neutral grey
    "SSC-Lasso":    "#1a9850",   # primary highlight - green
    "SSC-OMP":      "#f46d43",   # orange
    "Hierarchical": "#762a83",   # purple
    "Leiden":       "#4477AA",   # supplementary
    "HDBSCAN":      "#999933",   # supplementary
    "PROCLUS":      "#88CCEE",   # supplementary (axis-parallel)
    "GMM":          "#CC6677",   # supplementary (soft)
    "4C":           "#117733",   # supplementary (correlation)
    "ORCLUS":       "#DDCC77",   # supplementary
    "COPAC":        "#AA4499",   # supplementary
    "ERiC":         "#882255",   # supplementary
}


# Token taxonomy: Danish/English synonyms folded to one English label

CATEGORY_TO_ENGLISH = {
    "REPETITIVE": "Single-Token", "REPETITIV": "Single-Token",
    "SYNTACTIC": "Punctuation", "SYNTAKTISK": "Punctuation",
    "NUMERIC": "Numerical", "NUMERISK": "Numerical",
    "FUNCTION_WORDS": "Function Word", "FUNKTIONSORD": "Function Word",
    "FORMATTING": "Formatting", "FORMATERING": "Formatting",
    "CODE": "Code",
    "PROPER_NOUNS": "Proper Noun", "EGENNAVN/START": "Proper Noun",
    "SEMANTIC": "Semantic", "SEMANTISK": "Semantic",
    "CONTENT": "Content", "INDHOLD": "Content",
}


_RCPARAMS = {
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "Times", "Nimbus Roman",
                            "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset":    "stix",
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.titleweight":    "normal",
    "axes.labelsize":      11,
    "legend.fontsize":     9,
    "legend.frameon":      False,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "lines.linewidth":     1.6,
    "lines.markersize":    5,
    "grid.alpha":          0.3,
    "grid.linestyle":      ":",
}


def apply_style():
    """Apply the unified rcParams. Call once at the top of each figure script."""
    plt.rcParams.update(_RCPARAMS)


def savefig_both(path_png, **kwargs):
    """Save the current figure as both PNG and PDF.

    `path_png` should end in `.png`; the PDF is written next to it.
    """
    if not path_png.endswith(".png"):
        raise ValueError(f"savefig_both expects a .png path, got: {path_png}")
    plt.savefig(path_png, **kwargs)
    plt.savefig(path_png.replace(".png", ".pdf"), **kwargs)


def model_color(name):
    """Look up a model colour, with case-insensitive fallback."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    for k, v in MODEL_COLORS.items():
        if k.lower() == name.lower():
            return v
    return "#444444"


def method_color(name):
    """Look up a method colour, with case-insensitive fallback."""
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    for k, v in METHOD_COLORS.items():
        if k.lower() == name.lower():
            return v
    return "#888888"


def normalize_category(cat):
    """Map a Danish or English category label to its canonical English form."""
    return CATEGORY_TO_ENGLISH.get(cat, cat)
