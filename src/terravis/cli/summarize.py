"""CLI: tabulate `terravis-score` results, one column per model or image folder (`terravis-summarize`)."""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Any

# Result file names from eval_scores; an --image_dir run has no @model and gen_seed.
_RESULT_NAME = re.compile(r"t2i_scoring__raw__(?P<run>.+)__(?P<mode>.+?)__judge=(?P<judge>.+?)"
                          r"(?:__@(?P<model>.+)__gen_seed=(?P<seed>\d+))?\.json")
# Upstream soft_tifa_analysis.py's skills, in its order; it raises on any other.
_GENEVAL2_SKILLS = ("object", "attribute", "count", "position", "verb")


def mean_score(rows: list[dict[str, Any]], score_mode: str) -> tuple[str, int]:
    """(mean as "0.1234" or "n/a", number of numeric scores); n/a is excluded."""
    entries = [row.get(score_mode) for row in rows]
    numeric = [e["score"] for e in entries if isinstance(e, dict) and isinstance(e.get("score"), (int, float))]
    return (f"{sum(numeric) / len(numeric):.4f}" if numeric else "n/a"), len(numeric)


def unigenbench_official(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """prompt_type -> {"Overall": %, <primary dimension>: %}, as on the official leaderboards (short and
    long prompts are separate ones): a dimension pools its testpoints (upstream calculate_scores), and
    Overall averages the primary dimensions. n/a images are skipped, as upstream."""
    hits = defaultdict(lambda: defaultdict(list))
    for row in rows:
        entries = row["unigenbench_score"]["score_entries"]
        for testpoint, hit in zip(entries["testpoints"], entries.get("per_testpoint", [])):
            hits[entries["prompt_type"]][testpoint.split("-", 1)[0].strip()].append(hit)
    official = {}
    for prompt_type, dims in sorted(hits.items()):
        acc = {dim: 100 * sum(h) / len(h) for dim, h in sorted(dims.items())}
        official[prompt_type] = {"Overall": sum(acc.values()) / len(acc), **acc}
    return official


def geneval2_official(rows: list[dict[str, Any]]) -> dict[str, float]:
    """{"Overall": %, <skill>: %, "Atomicity=<n>": %}, as upstream evaluation.py and soft_tifa_analysis.py:
    Overall averages the prompts' Soft-TIFA GM (the official number), a skill pools its questions across
    prompts (Soft-TIFA AM), and an atomicity averages its prompts' Soft-TIFA GM."""
    by_skill, by_atoms = defaultdict(list), defaultdict(list)
    for row in rows:
        result = row["geneval2_score"]
        entries = result["score_entries"]
        for skill, p in zip(entries["skills"], entries["per_question"]):
            by_skill[skill].append(p)
        by_atoms[entries["atom_count"]].append(result["score"])
    scores = [s for group in by_atoms.values() for s in group]
    return {"Overall": 100 * sum(scores) / len(scores),
            **{s.capitalize(): 100 * sum(by_skill[s]) / len(by_skill[s])
               for s in sorted(by_skill, key=_GENEVAL2_SKILLS.index)},
            **{f"Atomicity={n}": 100 * sum(g) / len(g) for n, g in sorted(by_atoms.items())}}


def _print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    rows = [header, *rows]
    widths = [max(map(len, column)) for column in zip(*rows)]
    print(f"\n== {title} ==")
    for r in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip())


def summarize(save_path: str) -> None:
    save_path = os.path.abspath(save_path)   # titles name real folders even for "." or "raw"
    paths = sorted(glob.glob(os.path.join(glob.escape(save_path), "**", "t2i_scoring__raw__*.json"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No terravis-score results (t2i_scoring__raw__*.json) under {save_path}.")

    runs, seeds = [], defaultdict(set)
    for path in paths:
        m = _RESULT_NAME.fullmatch(os.path.basename(path))
        if m is None:
            continue
        raw_dir = os.path.dirname(path)
        if m["model"]:   # Step 2: a benchmark's models side by side
            group, column = raw_dir, m["model"]
        else:            # --image_dir: the folders scored into one save_path side by side
            group, column = os.path.dirname(os.path.dirname(raw_dir)), m["run"]
        runs.append((path, group, column, m["seed"], m["mode"], m["judge"]))
        seeds[group, column].add(m["seed"])

    means = defaultdict(lambda: defaultdict(dict))      # group -> (mode, judge) -> column -> cell
    official = defaultdict(lambda: defaultdict(dict))   # group -> (judge, title) -> column -> figures (%)
    for path, group, column, seed, mode, judge in runs:
        if len(seeds[group, column]) > 1:
            column = f"{column} seed={seed}"
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        mean, n = mean_score(rows, mode)
        means[group][mode, judge][column] = mean if n == len(rows) else f"{mean} ({n}/{len(rows)})"
        if mode == "unigenbench_score":
            for prompt_type, acc in unigenbench_official(rows).items():
                official[group][judge, f"UniGenBench++ leaderboard accuracy (%), {prompt_type} prompts"][column] = acc
        elif mode == "geneval2_score":
            official[group][judge, "GenEval 2 official scores (%): skills Soft-TIFA AM, the rest GM"][column] = (
                geneval2_official(rows))

    for group, metrics in means.items():
        columns = sorted({c for cells in metrics.values() for c in cells})
        _print_table(f"{group}: mean score (scored/total if any n/a)", ["metric [judge]", *columns],
                     [[f"{mode} [{judge}]", *(cells.get(c, "") for c in columns)]
                      for (mode, judge), cells in sorted(metrics.items())])
        for (judge, title), table in sorted(official[group].items()):
            columns = sorted(table)
            dims = list(dict.fromkeys(d for c in columns for d in table[c]))
            _print_table(f"{group}: {title} [{judge}]", ["dimension", *columns],
                         [[d, *(f"{table[c][d]:.2f}" if d in table[c] else "" for c in columns)] for d in dims])


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabulate terravis-score results: each metric's mean per model or image folder.")
    parser.add_argument("save_path", help="The --save_path given to terravis-score, or any folder under it.")
    summarize(parser.parse_args().save_path)


if __name__ == "__main__":
    main()
