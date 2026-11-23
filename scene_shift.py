"""Describe target-marginal transfer in the existing spatial holdouts."""
import csv
from pathlib import Path

from finite_study import inputs, np
from scipy.io import loadmat


def main():
    rows = []
    for scene, path in inputs().items():
        target = loadmat(path)[scene.lower() + "_corrected"][:, :, -16:].mean(axis=-1)
        strips = np.arange(target.shape[0]) * 4 // target.shape[0]
        for fold in range(4):
            reference = target[strips == (fold + 1) % 4]
            test_rows = np.flatnonzero(strips == (fold + 3) % 4)[::4]
            test = target[test_rows, ::4]
            standardized = (test - reference.mean()) / reference.std(ddof=1)
            rows.append(dict(scene=scene, fold=fold, test_pixels=test.size,
                             test_mean_in_reference_sd=standardized.mean(),
                             test_sd_in_reference_sd=standardized.std(ddof=1)))
    with Path("results/scene_shift.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
