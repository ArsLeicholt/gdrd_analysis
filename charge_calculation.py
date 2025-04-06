import csv
import argparse

def parse_fasta(file_path):
    """
    Generator that yields tuples (header, sequence) from a FASTA file.
    Assumes sequences may span multiple lines.
    """
    with open(file_path, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_lines)
                header = line[1:]  # remove the '>' character
                seq_lines = []
            else:
                seq_lines.append(line)
        # yield the last entry if exists
        if header is not None:
            yield header, ''.join(seq_lines)

def calculate_charge(sequence):
    """
    Calculate net charge of an amino acid sequence.
    Here, we assume:
      - Lysine (K), Arginine (R), and Histidine (H) each contribute +1.
      - Aspartic acid (D) and Glutamic acid (E) each contribute -1.
    """
    charge = 0
    for aa in sequence.upper():
        if aa in ('K', 'R', 'H'):
            charge += 1
        elif aa in ('D', 'E'):
            charge -= 1
    return charge

def main():
    parser = argparse.ArgumentParser(
        description="Calculate net charge of amino acid sequences from a FASTA file."
    )
    parser.add_argument("input", help="Input FASTA file")
    parser.add_argument("output", help="Output CSV file")
    args = parser.parse_args()

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Header", "Charge"])
        for header, sequence in parse_fasta(args.input):
            net_charge = calculate_charge(sequence)
            writer.writerow([header, net_charge])

if __name__ == "__main__":
    main()
