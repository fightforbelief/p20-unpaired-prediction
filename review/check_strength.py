"""External review of release 42aa864. Run from the P20 project root."""
import csv
import hashlib
import io
import math
import platform
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri

ALPHA = .05
Z = -ndtri(ALPHA / 2)
ONE = -ndtri(ALPHA)
DENSITY = math.exp(-Z*Z/2) / math.sqrt(2*math.pi)
ROOT = Path("results")
DATA = Path("incoming/levin09.zip")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def one_tail(d, w, quantile):
    # Solve the inner support with an independent bounded minimizer.
    def obj(lam):
        return ((d*w)**2 / (d*d + lam*(1-d*d))).sum() + quantile**2/(1-lam)
    fit = minimize_scalar(obj, bounds=(0., 1.), method="bounded", options={"xatol": 1e-13})
    if not fit.success:
        raise RuntimeError(fit.message)
    return math.sqrt(min(obj(0.), fit.fun))

def main():
    assert sha(DATA) == "c146576171f75335fe72388c9289350bcd5450ff882f8a525a6056232547e386"
    assert sha(ROOT/"conditional.csv") == "9a8e5b63e962feb8f7b5e4b8d1f81e9e4d7a6345b2c5c8fd159d9fc26ad9d9fa"
    with (ROOT/"conditional.csv").open() as file:
        ref = list(csv.DictReader(file))
    wrows = np.random.default_rng(20261123).standard_normal((16,1024))
    t = np.sin(np.pi*np.fft.fftfreq(32))**2
    prior = (1+64*(t[:,None]+t[None,:]))**-.5
    rows = []
    with zipfile.ZipFile(DATA) as kernels:
        for i in range(1,9):
            raw = kernels.read(f"levin09_kernels/levin09_kernel{i}.txt")
            kernel = np.loadtxt(io.BytesIO(raw))
            s = (abs(np.fft.fft2(kernel,s=(32,32),axes=(0,1)))*prior).ravel()
            d = np.sort(s/np.hypot(s,.2))[::-1]
            for j, w in enumerate(wrows):
                rec = ref[(i-1)*16+j]
                assert (int(rec['kernel']),int(rec['draw'])) == (i,j+1)
                B = max(Z, one_tail(d,w,ONE))
                # Upper rounding of the dual minimum is conservative for the interval.
                # For the theoretical gap bound, use a *feasible* mean/sd from the
                # release and M, both independent lower bounds for the exact B.
                floor = max(Z, np.linalg.norm(d*w),
                            float(rec['witness_mean'])+ONE*float(rec['witness_sd']))
                assert B >= floor-1e-11
                assert abs(one_tail(d,w,Z)-float(rec['union'])) < 1e-9
                eps = ndtr(-floor)
                slack = eps / DENSITY
                assert float(rec['lower']) <= B+slack+1e-10
                assert floor <= float(rec['upper'])+1e-10
                rows.append([i,j+1,B,floor,slack,B+slack,(1-float(rec["upper"])/float(rec["union"]))*100])
    print("python",platform.python_version(),"records",len(rows))
    print("B range",min(r[2] for r in rows),max(r[2] for r in rows))
    print("largest theoretical excess bound",max(r[4] for r in rows))
    print("largest numeric lower/upper separation including optimization",max(r[5]-r[3] for r in rows))
    print("median and range percent gain versus equal-tail",np.median([r[-1] for r in rows]),min(r[-1] for r in rows),max(r[-1] for r in rows))
    print("first case",rows[0])
    out = Path(__file__).with_name("baseline-audit.csv")
    with out.open("w", newline="") as f:
        wr=csv.writer(f,lineterminator="\n")
        wr.writerow(["kernel","draw","one_tail_radius","feasible_floor","analytic_excess_bound","valid_radius","original_gain_pct"])
        wr.writerows(rows)

if __name__ == "__main__":
    main()
