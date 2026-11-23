"""Audit what batch alignment alone explains. Run from the project root."""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import hadamard
from finite_design import allocate, predict_axes
from replay import compare

OUT = Path("results/span/comparison.csv")


def evaluate(path):
    p,m,n,reps = 64,16,256,256
    rows=[]
    for case,frac in [("flat",np.ones(m)/m),("concentrated",np.r_[.75,np.full(m-1,.25/(m-1))])]:
        # Orthogonal columns with the chosen spectrum and constant row norms.
        g=np.zeros((m,p))
        g[:,:m]=hadamard(m)*np.sqrt(p*frac)
        d=np.sqrt(frac/frac.max())
        schedules=dict(batch=allocate(d,n), mean_info=allocate(d,n,"mean_info"),
                       span_equal=np.arange(n)%m, reusable=np.arange(n)%p)
        if case=="flat":
            assert np.all(np.bincount(schedules["batch"])==n//m)
        for policy,idx in schedules.items():
            counts=np.bincount(idx,minlength=p)[:m]
            expected=sum(weight*quad(lambda t: math.exp(-t)*(1+2*t)**(-k/2),
                                     0,np.inf)[0] for k,weight in zip(counts,p*frac))
            # Match archives across policies even when counts differ.
            stream=np.random.default_rng(20261201+(case=="concentrated"))
            halves=[]; covered=[]
            for _ in range(reps):
                x=stream.normal(size=(n,p))
                t=.8*x.sum(axis=1)/np.sqrt(p)+.6*stream.normal(size=n)
                y=.8*g.sum(axis=1)/np.sqrt(p)+.6*stream.normal(size=m)
                pred,half,_=predict_axes(idx,x[np.arange(n),idx],t,g)
                halves.append(np.mean(half**2))
                covered.extend(abs(y-pred)<=half)
            rows.append(dict(case=case,policy=policy,budget=n,queries=m,
                             leading_count=counts[0],expected_modulus2=expected,
                             rms_half=np.sqrt(np.mean(halves)),coverage=np.mean(covered)))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="--replay":
        replay=Path("incoming/span-replay.csv")
        evaluate(replay)
        compare(OUT,replay)
        print("span control replay verified")
    else:
        OUT.parent.mkdir(parents=True,exist_ok=True)
        evaluate(OUT)
