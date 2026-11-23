"""Replay from the project root without replacing the reference results."""
import csv
import json
import math
import tempfile
from pathlib import Path

import study


def compare(reference, replay):
    with reference.open(newline="") as f, replay.open(newline="") as g:
        left, right = list(csv.reader(f)), list(csv.reader(g))
    if not left or len(left) != len(right) or left[0] != right[0]:
        raise ValueError(f"schema or row count changed in {reference}")
    for i, (row, other) in enumerate(zip(left[1:], right[1:]), 2):
        if len(row) != len(other):
            raise ValueError(f"column count changed on row {i} in {reference}")
        for a, b in zip(row, other):
            if a == b:
                continue
            try:
                equal = math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)
            except ValueError:
                equal = False
            if not equal:
                raise ValueError(f"{reference} row {i}: {a} versus {b}")


if __name__ == "__main__":
    reference = Path("results")
    record = json.loads((reference / "run.json").read_text())
    for group, base in [("sha256", Path(".")), ("results", reference)]:
        for name, expected in record[group].items():
            if study.digest(base / name) != expected:
                raise ValueError(f"reference hash mismatch: {name}")
    study.audit()
    study.OUT = Path(tempfile.mkdtemp(prefix="replay-", dir="incoming"))
    study.run()
    for name, expected in record["results"].items():
        compare(reference / name, study.OUT / name)
        result = "byte-identical" if study.digest(study.OUT / name) == expected else "within tolerance"
        print(name, result)
    print("Replay verified:", study.OUT)
