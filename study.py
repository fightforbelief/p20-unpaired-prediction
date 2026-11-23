"""Reproduce all numerical results. Run from the project root."""
import csv
import hashlib
import io
import json
import os
import platform
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import scipy
from scipy.special import ndtr, ndtri

from intervals import sharp_radius, support

ARCHIVE = Path("incoming/levin09.zip")
URL = "https://cabouman.github.io/grad_labs/labs/resources/levin09_kernels.zip"
HASH = "c146576171f75335fe72388c9289350bcd5450ff882f8a525a6056232547e386"
OUT = Path("results")
LEVEL = .05
Z = -ndtri(LEVEL / 2)
SEED = 20261123


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit():
    if not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(exist_ok=True)
        with urlopen(URL, timeout=60) as response:
            blob = response.read()
        if hashlib.sha256(blob).hexdigest() != HASH:
            raise ValueError("archive changed at source")
        ARCHIVE.write_bytes(blob)
    if digest(ARCHIVE) != HASH:
        raise ValueError("unexpected input checksum")
    print(platform.python_version(), np.__version__, scipy.__version__, HASH, flush=True)


def run():
    audit()
    OUT.mkdir(exist_ok=True)
    grid = np.sin(np.pi * np.fft.fftfreq(32))**2
    prior = 1 / np.sqrt(1 + 64 * (grid[:, None] + grid[None, :]))
    rng = np.random.default_rng(SEED)
    draws = rng.standard_normal((16, 1024))
    operators, exact = [], []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for index in range(1, 9):
            kernel = np.loadtxt(io.BytesIO(archive.read(f"levin09_kernels/levin09_kernel{index}.txt")))
            assert np.isclose(kernel.sum(), 1) and kernel.min() >= 0
            blur = abs(np.fft.fft2(kernel, s=(32, 32), axes=(0, 1)))
            for model, spectrum in [('smooth', prior), ('white', np.ones_like(prior))]:
                for noise in [.05, .2, .8]:
                    s = (blur * spectrum).ravel()
                    d = np.sort(s / np.hypot(s, noise))[::-1]
                    trace = d @ d
                    tail = d[32:] @ d[32:]
                    operators.append(dict(kernel=index, prior=model, noise=noise,
                                          effective_dimension=trace, top32_tail=tail,
                                          random32_mean_tail=(1-32/len(d))*trace))
                    if model == 'smooth' and noise == .2:
                        for j, w in enumerate(draws):
                            g = d*w
                            result = sharp_radius(d, w, tol=.0005)
                            union, _ = support(d, g, 1., Z)
                            u = result.coupling
                            m, sd = g @ u, np.sqrt(1 - np.sum((d*u)**2))
                            coverage = ndtr((result.upper-m)/sd) - ndtr((-result.upper-m)/sd)
                            assert .95-1e-11 <= coverage <= .951 and result.upper <= union+.0005
                            exact.append(dict(kernel=index, draw=j+1, mean_radius=np.linalg.norm(g),
                                              lower=result.lower, upper=result.upper, union=union,
                                              witness_mean=m, witness_sd=sd,
                                              witness_coverage=coverage))
            print("finished kernel", index, flush=True)
    save("operators.csv", operators)
    save("conditional.csv", exact)

    # Conditional simulation. Each independent row mimics new calibrators and a
    # new target at the fixed test observation. Spectral coordinates are sorted.
    bridge = []
    index, w = 1, draws[0]
    with zipfile.ZipFile(ARCHIVE) as archive:
        kernel = np.loadtxt(io.BytesIO(archive.read("levin09_kernels/levin09_kernel1.txt")))
    s = (abs(np.fft.fft2(kernel, s=(32,32), axes=(0,1))) * prior).ravel()
    d = np.sort(s / np.hypot(s,.2))[::-1]
    g = d*w
    radius = np.linalg.norm(g)
    direction = g / radius
    runs = 50000
    for n in [4,16,64,256]:
        for b in [0., .5, .95]:
            energy = rng.chisquare(n, size=runs)
            bhat = b + np.sqrt((1-b*b)/energy)*rng.standard_normal(runs)
            sd = np.sqrt(1 - np.sum((d*direction*b)**2))
            target = radius*b + sd*rng.standard_normal(runs)
            half = Z*np.sqrt(1 + radius**2/energy)
            err = abs(target-radius*bhat)
            bridge.append(dict(kernel=index, draw=1, n=n, b=b, replicates=runs,
                               median_half=np.median(half), covered=int(np.sum(err<=half)),
                               naive_covered=int(np.sum(err<=Z))))
    save("paired.csv",bridge)
    files = ["study.py","intervals.py","test_intervals.py","requirements.txt","protocol.md"]
    report = dict(platform=platform.platform(), python=sys.version, numpy=np.__version__,
                  scipy=scipy.__version__, seed=SEED, input_sha256=HASH, pid=os.getpid(),
                  command="python study.py", sha256={f:digest(f) for f in files},
                  results={p.name:digest(p) for p in sorted(OUT.glob("*.csv"))})
    (OUT/"run.json").write_text(json.dumps(report, indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["--audit"]:
        audit()
    else:
        run()
