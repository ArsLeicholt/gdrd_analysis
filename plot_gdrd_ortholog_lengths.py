import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("gdrd_orthologs.csv", header=1)

    labels = df["Species"].astype(str).tolist()
    n = len(labels)
    group_gap = 1.6
    x = np.arange(n) * group_gap
    width = 0.3
    spacing = width * 1.1

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.bar(
        x - 1.5 * spacing,
        df["full protein"],
        width,
        label="Full protein",
        color="#808080",
        alpha=0.8,
        zorder=3,
    )
    ax.bar(
        x - 0.5 * spacing,
        df["N-terminus"],
        width,
        label="N-terminus",
        color="#F58518",
        zorder=3,
    )
    ax.bar(
        x + 0.5 * spacing,
        df["central helix"],
        width,
        label="Central helix",
        color="#54A24B",
        zorder=3,
    )
    ax.bar(
        x + 1.5 * spacing,
        df["C-terminus"],
        width,
        label="C-terminus",
        color="#4C78A8",
        zorder=3,
    )

    group_abbrev = {
        "melanogaster": "mel",
        "obscura": "obsc",
        "repleta": "rep",
        "virilis": "vir",
        "robusta": "rob",
        "melanica": "mela",
        "Hawaiian ": "haw",
        "immigrans": "imm",
        "cardini": "etc.",
        "funebris": "etc.",
        "quinaria": "etc.",
    }

    group_values = df["Group"].astype(str).tolist()
    line_y = -0.04
    text_y = -0.09
    gap = group_gap * 0.15
    spans = []
    start = 0
    for i in range(1, n + 1):
        if i == n or group_values[i] != group_values[start]:
            end = i - 1
            group_name = group_values[start].strip()
            abbrev = group_abbrev.get(group_name, group_name[:4].lower())
            spans.append((start, end, abbrev))
            start = i

    merged_spans = []
    for start, end, abbrev in spans:
        if merged_spans and merged_spans[-1][2] == abbrev:
            merged_spans[-1] = (merged_spans[-1][0], end, abbrev)
        else:
            merged_spans.append((start, end, abbrev))

    axis_transform = ax.get_xaxis_transform()
    for start, end, abbrev in merged_spans:
        left = x[start] - group_gap / 2
        right = x[end] + group_gap / 2
        center = (x[start] + x[end]) / 2
        ax.hlines(
            line_y,
            left + gap,
            right - gap,
            color="black",
            linewidth=1.2,
            transform=axis_transform,
            clip_on=False,
        )
        ax.text(
            center,
            text_y,
            abbrev,
            ha="center",
            va="top",
            fontsize=14,
            transform=axis_transform,
        )

    ax.set_ylim(0, 250)
    ax.set_ylabel("Length (aa)", fontsize=14)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels([""] * n)
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
    ax.legend(fontsize=14)
    ax.set_title(
        "Length differences between Gdrd orthologs protein sequences",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig("gdrd_orthologs_length_differences.png", dpi=300)


if __name__ == "__main__":
    main()
