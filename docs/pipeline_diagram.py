"""
Generate docs/pipeline.png — a colorful serpentine diagram of the DataPilot AI
LangGraph pipeline. Pure matplotlib (no network, no emoji fonts needed) so it
renders identically anywhere.

    python docs/pipeline_diagram.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

HALFW, HALFH = 1.3, 0.62
COLX = {0: 0.0, 1: 3.3, 2: 6.6, 3: 9.9, 4: 13.2}
ROWY = {0: 5.0, 1: 2.5, 2: 0.0}

# class -> (face, text, border)
STYLE = {
    "se": ("#111827", "#ffffff", "#000000"),
    "prep": ("#bfdbfe", "#0b3a8f", "#3b82f6"),
    "decide": ("#fde68a", "#7c4a03", "#f59e0b"),
    "plan": ("#fed7aa", "#7c2d12", "#fb923c"),
    "exec": ("#bbf7d0", "#14532d", "#22c55e"),
    "deploy": ("#e9d5ff", "#581c87", "#a855f7"),
}

# id -> (col, row, label, class)
NODES = {
    "S": (0, 0, "START", "se"),
    "IP": (1, 0, "Input\nPreparation", "prep"),
    "PD": (2, 0, "Problem\nDefinition\n(human pick)", "decide"),
    "PP": (3, 0, "Preprocess\n(review)", "plan"),
    "PPX": (4, 0, "Execute", "exec"),
    "EDA": (4, 1, "EDA\n(review)", "plan"),
    "EDAX": (3, 1, "Execute\ncharts", "exec"),
    "FE": (2, 1, "Feature Eng\n(review)", "plan"),
    "FEX": (1, 1, "Execute", "exec"),
    "MOD": (0, 1, "Modeling\n(review)", "plan"),
    "TRX": (0, 2, "Train\n→ model.pkl", "exec"),
    "EV": (1, 2, "Evaluation\n(review)", "plan"),
    "EVX": (2, 2, "Execute\nmetrics", "exec"),
    "DEP": (3, 2, "Deployment\n(confirm)", "deploy"),
    "E": (4, 2, "END", "se"),
}

FORWARD = [
    ("S", "IP"), ("IP", "PD"), ("PD", "PP"), ("PP", "PPX"), ("PPX", "EDA"),
    ("EDA", "EDAX"), ("EDAX", "FE"), ("FE", "FEX"), ("FEX", "MOD"), ("MOD", "TRX"),
    ("TRX", "EV"), ("EV", "EVX"), ("EVX", "DEP"), ("DEP", "E"),
]
RETRY = [("PPX", "PP"), ("EDAX", "EDA"), ("FEX", "FE"), ("TRX", "MOD"), ("EVX", "EV")]


def center(nid):
    col, row, _, _ = NODES[nid]
    return COLX[col], ROWY[row]


def edge_point(c, toward):
    dx, dy = toward[0] - c[0], toward[1] - c[1]
    if dx == 0 and dy == 0:
        return c
    sx = HALFW / abs(dx) if dx else 1e9
    sy = HALFH / abs(dy) if dy else 1e9
    t = min(sx, sy)
    return c[0] + dx * t, c[1] + dy * t


def main():
    fig, ax = plt.subplots(figsize=(15, 7.2))

    for nid, (col, row, label, cls) in NODES.items():
        x, y = COLX[col], ROWY[row]
        face, txt, border = STYLE[cls]
        ax.add_patch(
            FancyBboxPatch(
                (x - HALFW, y - HALFH), 2 * HALFW, 2 * HALFH,
                boxstyle="round,pad=0.02,rounding_size=0.2",
                linewidth=2, edgecolor=border, facecolor=face, zorder=2,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
                color=txt, fontweight="bold", zorder=3)

    for a, b in FORWARD:
        ca, cb = center(a), center(b)
        ax.add_patch(FancyArrowPatch(
            edge_point(ca, cb), edge_point(cb, ca),
            arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#334155", zorder=1,
        ))

    for a, b in RETRY:
        ca, cb = center(a), center(b)
        ax.add_patch(FancyArrowPatch(
            edge_point(ca, cb), edge_point(cb, ca),
            arrowstyle="-|>", mutation_scale=13, lw=1.7, color="#dc2626",
            linestyle="--", connectionstyle="arc3,rad=-0.45", zorder=1,
        ))

    ax.set_title("DataPilot AI — pipeline flow", fontsize=16, fontweight="bold", pad=14)

    legend_handles = [
        mpatches.Patch(facecolor=STYLE["prep"][0], edgecolor=STYLE["prep"][2], label="Input prep (no LLM)"),
        mpatches.Patch(facecolor=STYLE["decide"][0], edgecolor=STYLE["decide"][2], label="Human decision"),
        mpatches.Patch(facecolor=STYLE["plan"][0], edgecolor=STYLE["plan"][2], label="LLM plan + human review"),
        mpatches.Patch(facecolor=STYLE["exec"][0], edgecolor=STYLE["exec"][2], label="Execute approved code"),
        mpatches.Patch(facecolor=STYLE["deploy"][0], edgecolor=STYLE["deploy"][2], label="Deployment"),
        Line2D([0], [0], color="#334155", lw=2.2, label="approved -> next phase"),
        Line2D([0], [0], color="#dc2626", lw=1.7, ls="--", label="error -> regenerate (self-heal)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=4, frameon=False, fontsize=9)

    ax.set_xlim(-2, 15.2)
    ax.set_ylim(-1.6, 6.4)
    ax.axis("off")
    plt.tight_layout()
    # SVG is the scalable artifact embedded in the README; PNG is a raster fallback.
    plt.savefig("docs/pipeline.svg", bbox_inches="tight", facecolor="white")
    plt.savefig("docs/pipeline.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote docs/pipeline.svg and docs/pipeline.png")


if __name__ == "__main__":
    main()
