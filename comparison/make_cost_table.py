# -*- coding: utf-8 -*-
"""
Assemble the consolidated computational-cost + memory table (reviewer R2-C7) from
the three measurement outputs, so the paper's table is reproducible and never
hand-copied:

    results/cost_table.md            (cost_benchmark.py)      -> extra NMF, training, inference
    results/cost_memory.md           (cost_memory.py)         -> peak RAM, model size
    results/cost_timing_breakdown.md (cost_memory.py --breakdown) -> preprocessing, NMF split

It lifts the already-formatted cell strings (no re-parsing of units, no new
numbers introduced) and rearranges them into one table whose columns map 1:1 to
the reviewer's request: preprocessing, NMF computation, training, inference,
memory -- for each approach. Writes results/cost_table_final.md.

    python comparison/make_cost_table.py

To regenerate the underlying measurements first (on a new machine):
    python comparison/cost_benchmark.py
    python comparison/cost_memory.py
    python comparison/cost_memory.py --breakdown
    python comparison/make_cost_table.py
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CANON = ["nmf", "cnn", "roi", "ch"]
DISPLAY = {"nmf": "NMF-only", "cnn": "CNN-only",
           "roi": "NMF+CNN (ROI)", "ch": "NMF+CNN (channel fusion)"}


def canon(name):
    """Map any file's arm label to a canonical key (order matters)."""
    n = name.lower()
    if "roi" in n:
        return "roi"
    if "channel" in n:          # "channels" or "channel fusion"
        return "ch"
    if "cnn-only" in n:
        return "cnn"
    if "nmf-only" in n:
        return "nmf"
    return None


def read_table_rows(path):
    """Return the '|'-delimited data rows (list of cell lists) of the FIRST table."""
    if not os.path.exists(path):
        raise SystemExit(f"missing input: {path}\nrun the measurement script first.")
    rows, in_table = [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("|"):
                in_table = True
                if set(s) <= set("|-: "):         # separator row
                    continue
                rows.append([c.strip() for c in s.strip("|").split("|")])
            elif in_table and not s.startswith("|"):
                break                              # table ended
    return rows                                    # rows[0] is the header


def by_arm(rows):
    """{canonical arm -> [cells]} from a table's data rows (skipping the header)."""
    out = {}
    for cells in rows[1:]:
        k = canon(cells[0])
        if k:
            out[k] = cells
    return out


def strip_time(cell):
    """'ROI localization: 9.1 s' -> '9.1 s'; '- (uses ...)' / '-' -> ''."""
    if not cell or cell.lstrip().startswith("-"):
        return ""
    return cell.split(":", 1)[1].strip() if ":" in cell else cell.strip()


def breakdown_value(rows, needle):
    for cells in rows:
        if needle.lower() in cells[0].lower() and "total" not in cells[0].lower():
            return cells[1].replace("*", "").strip()
    raise SystemExit(f"could not find '{needle}' in the breakdown table")


def model_size_line(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "model size" in line.lower():
                return line.strip().lstrip("*").replace("**", "").strip()
    return ""


def main():
    cost = by_arm(read_table_rows(os.path.join(RESULTS, "cost_table.md")))
    mem = by_arm(read_table_rows(os.path.join(RESULTS, "cost_memory.md")))
    brk = read_table_rows(os.path.join(RESULTS, "cost_timing_breakdown.md"))

    for k in CANON:                                # fail loudly if an arm is missing
        if k not in cost or k not in mem:
            raise SystemExit(f"arm '{k}' not found in cost_table.md/cost_memory.md")

    prep = breakdown_value(brk, "Preprocessing")           # e.g. '16.9 s'
    nmf_only_nmf = breakdown_value(brk, "NMF feature")      # e.g. '17.4 s'

    def nmf_cell(k):
        if k == "nmf":
            return nmf_only_nmf                            # NMF-only's feature NMF
        if k == "cnn":
            return "–"                                     # no NMF
        return strip_time(cost[k][1]) or "–"               # hybrids: ROI/channel map NMF

    def ram_cell(k):
        return f"{mem[k][1]} / {mem[k][2]}"                # 'train / infer'

    def train_cell(k):
        return cost[k][2].replace("->", "→")

    def arrow(s):
        return s.replace("->", "→")

    lines = [
        "# Table 5 (consolidated): computational cost and memory per approach (CPU-only)\n",
        "Auto-assembled by make_cost_table.py from cost_table.md, cost_memory.md and "
        "cost_timing_breakdown.md. Columns map 1:1 to the reviewer's request.\n",
        "| Approach | Preprocessing (per vol) | NMF computation (per vol) | Training | "
        "Inference (per vol) | Peak RAM (training / inference) |",
        "|---|---|---|---|---|---|",
    ]
    for k in CANON:
        lines.append(
            f"| {DISPLAY[k]} | {prep} | {nmf_cell(k)} | {train_cell(k)} | "
            f"{cost[k][3]} | {ram_cell(k)} |")

    lines += [
        "",
        f"Preprocessing (body mask, N4, denoising, normalization) is shared by all "
        f"approaches, computed once per volume and cached (~{prep}). NMF computation is "
        f"one-time and cached: the NMF-only pipeline's feature NMF (~{nmf_only_nmf}, "
        "max_iter=1000) versus the lighter NMF the hybrids use for the ROI/channel maps "
        "(max_iter=200). Training and inference times are extrapolated/averaged as in "
        "cost_table.md. Peak RAM is the peak resident-set size (NMF-only measured without "
        "the PyTorch stack).",
        "",
        model_size_line(os.path.join(RESULTS, "cost_memory.md")),
        "",
    ]
    out = "\n".join(lines) + "\n"
    path = os.path.join(RESULTS, "cost_table_final.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
