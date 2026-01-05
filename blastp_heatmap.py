import os
import subprocess
import tempfile

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Bio import SeqIO

BLASTP = "/path/"
MAKEBLASTDB = "/path/"

ORDER = ["mel", "sim", "yak", "ana", "moj", "vir", "gri"]


def normalize_id(raw_id: str) -> str:
    return raw_id.lstrip(">").strip()


def infer_abbrev(seq_id: str) -> str:
    lower = seq_id.lower()
    for abbr in ORDER:
        if abbr in lower:
            return abbr
    raise ValueError(f"Cannot infer abbreviation from id: {seq_id}")


def build_id_map(fasta_path: str) -> dict:
    id_map = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        desc_id = normalize_id(record.description)
        try:
            abbr = infer_abbrev(desc_id)
        except ValueError:
            abbr = infer_abbrev(normalize_id(record.id))
        if desc_id in id_map and id_map[desc_id] != abbr:
            raise ValueError(f"Conflicting mapping for {desc_id}: {id_map[desc_id]} vs {abbr}")
        id_map[desc_id] = abbr
    missing = [abbr for abbr in ORDER if abbr not in id_map.values()]
    if missing:
        raise ValueError(f"Missing abbreviations in {fasta_path}: {missing}")
    return id_map


def run_blastp(fasta_path: str, id_map: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_fasta = os.path.join(tmpdir, "query.fasta")
        seen_abbr = set()
        records = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            desc_id = normalize_id(record.description)
            abbr = id_map.get(desc_id)
            if abbr is None:
                abbr = infer_abbrev(desc_id)
            if abbr in seen_abbr:
                raise ValueError(f"Duplicate abbreviation {abbr} in {fasta_path}")
            seen_abbr.add(abbr)
            record.id = abbr
            record.description = ""
            records.append(record)
        SeqIO.write(records, temp_fasta, "fasta")

        db_prefix = os.path.join(tmpdir, "blastdb")
        subprocess.run(
            [
                MAKEBLASTDB,
                "-in",
                temp_fasta,
                "-dbtype",
                "prot",
                "-out",
                db_prefix,
            ],
            check=True,
        )
        out_path = os.path.join(tmpdir, "blast.tsv")
        subprocess.run(
            [
                BLASTP,
                "-task",
                "blastp-short",
                "-query",
                temp_fasta,
                "-db",
                db_prefix,
                "-out",
                out_path,
                "-outfmt",
                "6 qseqid sseqid nident length bitscore",
                "-evalue",
                "0.05",
                "-max_target_seqs",
                "100",
            ],
            check=True,
        )

        best = {}
        with open(out_path, "r", encoding="utf-8") as handle:
            for line in handle:
                qseqid, sseqid, nident, length, bitscore = line.rstrip("\n").split("\t")
                qabbr = normalize_id(qseqid)
                sabbr = normalize_id(sseqid)
                nident = int(nident)
                length = int(length)
                bitscore = float(bitscore)
                key = (qabbr, sabbr)
                if key not in best or bitscore > best[key]["bitscore"]:
                    pct = (nident / length * 100.0) if length else 0.0
                    best[key] = {
                        "nident": nident,
                        "length": length,
                        "pct": pct,
                        "bitscore": bitscore,
                    }
        return best


def build_matrix(best_hits: dict) -> tuple[np.ndarray, list[list[tuple[int, int, float]]]]:
    n = len(ORDER)
    matrix = np.zeros((n, n), dtype=float)
    text_values: list[list[tuple[int, int, float]]] = [
        [(0, 0, 0.0) for _ in range(n)] for _ in range(n)
    ]
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            ab = best_hits.get((a, b))
            if ab is None:
                matrix[i, j] = 0.0
                text_values[i][j] = (0, 0, 0.0)
                continue
            matrix[i, j] = ab["pct"]
            text_values[i][j] = (ab["nident"], ab["length"], ab["pct"])
    return matrix, text_values


def plot_half_heatmap(
    matrix: np.ndarray, text_values: list[list[tuple[int, int, float]]], title: str, out_path: str
) -> None:
    n = matrix.shape[0]
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    masked = np.ma.array(matrix, mask=mask)

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(8.4, 7))
    cmap = plt.cm.Greys
    cmap.set_bad(color="white")

    ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, origin="lower")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(ORDER, rotation=45, ha="right")
    ax.set_yticklabels(ORDER)

    for i in range(n):
        for j in range(n):
            if i >= j:
                nident, length, pident = text_values[i][j]
                pident_floor = math.floor(pident * 10) / 10 if pident >= 0 else 0.0
                text_color = "white" if pident >= 50 else "black"
                ax.text(
                    j,
                    i + 0.12,
                    f"{nident}/{length}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=14,
                )
                ax.text(
                    j,
                    i - 0.12,
                    f"({pident_floor:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                )

    ax.set_title(title, fontsize=8, fontweight="bold", pad=12)
    ax.tick_params(labelsize=13)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def process(fasta_path: str, title: str, out_path: str) -> None:
    id_map = build_id_map(fasta_path)
    best_hits = run_blastp(fasta_path, id_map)
    matrix, text_values = build_matrix(best_hits)
    plot_half_heatmap(matrix, text_values, title, out_path)


def build_combined_figure(
    image_paths: list[str], panel_labels: list[str], title: str, out_path: str
) -> None:
    fig = plt.figure(figsize=(5030 / 300, 4459 / 300), dpi=300)
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]
    for ax, path, label in zip(axes, image_paths, panel_labels):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.annotate(
            label,
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            xytext=(0, 5),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            color="black",
        )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01, wspace=0.02, hspace=0.02)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    process(
        "gdrd_orthologs_for_analysis.fasta",
        "Gdrd orthologs (tested): All-against-all PBLAST sequence identities (full sequences)",
        "all_against_all_pblast_full_sequences.png",
    )
    process(
        "gdrd_C_termini.fasta",
        "Gdrd orthologs (tested): All-against-all PBLAST sequence identities (C-termini)",
        "all_against_all_pblast_c_termini.png",
    )
    process(
        "gdrd_N_termini.fasta",
        "Gdrd orthologs (tested): All-against-all PBLAST sequence identities (N-termini)",
        "all_against_all_pblast_n_termini.png",
    )
    process(
        "gdrd_central_helix.fasta",
        "Gdrd orthologs (tested): All-against-all PBLAST sequence identities (central helix)",
        "all_against_all_pblast_central_helix.png",
    )
    build_combined_figure(
        [
            "all_against_all_pblast_full_sequences.png",
            "all_against_all_pblast_c_termini.png",
            "all_against_all_pblast_n_termini.png",
            "all_against_all_pblast_central_helix.png",
        ],
        ["A", "B", "C", "D"],
        "Gdrd orthologs (tested): All-against-all PBLAST sequence identities",
        "all_against_all_pblast_combined.png",
    )


if __name__ == "__main__":
    main()
