import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO


NEGATIVE = {"D", "E"}
POSITIVE = {"K", "R", "H"}


def normalize_species(name: str) -> str:
    return name.replace(" ", "").strip()


def build_sequence_map(fasta_path: str) -> dict[str, str]:
    seq_map = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_map[normalize_species(record.id)] = str(record.seq)
    return seq_map


def sequence_charge(seq: str) -> int:
    charge = 0
    for aa in seq:
        if aa in NEGATIVE:
            charge -= 1
        elif aa in POSITIVE:
            charge += 1
    return charge


def find_sequence(species: str, seq_map: dict[str, str]) -> str:
    key = normalize_species(species)
    if key in seq_map:
        return seq_map[key]
    if key.startswith("S."):
        alt = f"D.{key[2:]}"
        if alt in seq_map:
            return seq_map[alt]
    return ""


def main() -> None:
    df = pd.read_csv("gdrd_orthologs.csv", header=1)

    n_seq_map = build_sequence_map("gdrd_orthologs_N_terminus.fasta")
    c_seq_map = build_sequence_map("gdrd_orthologs_C_terminus.fasta")

    labels = df["Species"].astype(str).tolist()
    n_charges = []
    c_charges = []
    missing = []
    for species in labels:
        n_seq = find_sequence(species, n_seq_map)
        c_seq = find_sequence(species, c_seq_map)
        if not n_seq or not c_seq:
            missing.append(species)
            continue
        n_charges.append(sequence_charge(n_seq))
        c_charges.append(sequence_charge(c_seq))

    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing sequence data for: {missing_list}")

    n = len(labels)
    group_gap = 1.6
    x = np.arange(n) * group_gap

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(
        x,
        n_charges,
        color="#F58518",
        marker="o",
        linewidth=2,
        label="N-terminus",
        zorder=3,
    )
    ax.plot(
        x,
        c_charges,
        color="#4C78A8",
        marker="o",
        linewidth=2,
        label="C-terminus",
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

    y_min = min(min(n_charges), min(c_charges))
    y_max = max(max(n_charges), max(c_charges))
    padding = max(1, int(round((y_max - y_min) * 0.1)))
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_ylabel("Charge", fontsize=14)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels([""] * n)
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
    ax.legend(fontsize=14)
    ax.set_title(
        "N- and C-terminus charge differences between Gdrd orthologs",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig("gdrd_orthologs_charge_termini.png", dpi=300)


if __name__ == "__main__":
    main()
