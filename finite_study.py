"""Run the predeclared finite-archive study from the P20 project root."""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.special import ndtr, ndtri

import study
from finite_design import allocate, folded_radius, predict_axes, predict_scores
from intervals import sharp_radius, support
from replay import compare

SEED = 20261124
POLICIES = ("greedy","proportional","balanced","top4")
REF = Path("results/finite")
SCENES = {
    "Indian_pines": ("https://raw.githubusercontent.com/ZhangQuan-hub/HyperspectralDatasets/main/Indian_pines_corrected.mat",
                     "ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939"),
    "Salinas": ("https://raw.githubusercontent.com/roman-permyakov/Salinas-Hyperspectral/HEAD/Salinas_corrected.mat",
                "5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d"),
}


def inputs():
    study.audit()
    paths = {}
    for scene,(url,expected) in SCENES.items():
        path = Path("incoming") / f"p20-{scene}_corrected.mat"
        if not path.exists():
            with urlopen(url,timeout=120) as response:
                blob = response.read()
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError(f"source changed for {scene}")
            path.write_bytes(blob)
        if study.digest(path) != expected:
            raise ValueError(f"incorrect cached scene {scene}")
        paths[scene] = path
    return paths


def save(folder,name,rows):
    with (folder/f"{name}.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary(y, pred, half, modulus2):
    return dict(queries=len(y), half50=np.median(half), half90=np.quantile(half,.9),
                coverage=np.mean(abs(y-pred)<=half), rmse=np.sqrt(np.mean((y-pred)**2)),
                mean_modulus2=np.mean(modulus2))


def limits(rng):
    rows = []
    for p in [8,32,128]:
        for n in [p,4*p,p*p]:
            g = rng.normal(size=(512,p))
            counts = np.bincount(np.arange(n)%p, minlength=p)
            e = rng.chisquare(counts, size=g.shape)
            for mode in ["reusable","query"]:
                if mode=="query":
                    signal = np.linalg.norm(g,axis=1,keepdims=True)
                    energy = rng.chisquare(n,size=(len(g),1))
                else:
                    signal,energy=g,e
                residual = signal/(1+energy)
                v = np.sum(residual**2 * energy,axis=1)
                b = np.sum(residual**2,axis=1)
                rows.append(dict(p=p,budget=n,mode=mode,root_mean_modulus2=np.sqrt(np.mean(v+b)),
                                 mean_half=folded_radius(np.sqrt(b),np.sqrt(1+v)).mean()))
    return rows


def tails():
    rows = []
    for d in [.5,1.]:
        for w in [0.,.75,1.5,2.25,4.,8.]:
            g=np.array([d*w])
            gain=np.array([d])
            sharp=sharp_radius(gain,[w],tol=1e-6)
            one,_=support(gain,g,1.,-ndtri(.05))
            union,_=support(gain,g,1.,-ndtri(.025))
            floor=max(one,-ndtri(.025))
            correction=ndtr(-floor)*np.sqrt(2*np.pi)*np.exp(ndtri(.025)**2/2)
            simple=min(union,floor+correction)
            assert sharp.lower <= simple+1e-10 and sharp.upper >= floor-1e-10
            rows.append(dict(gain=d,w=w,sharp_lower=sharp.lower,sharp_upper=sharp.upper,
                             union=union,simple=simple))
    return rows


def camera(rng):
    rows = []
    grid = np.sin(np.pi*np.fft.fftfreq(32))**2
    prior = 1/np.sqrt(1+64*(grid[:,None]+grid[None,:]))
    with zipfile.ZipFile(study.ARCHIVE) as archive:
        for index in range(1,9):
            k = np.loadtxt(io.BytesIO(archive.read(f"levin09_kernels/levin09_kernel{index}.txt")))
            s = (abs(np.fft.fft2(k,s=(32,32),axes=(0,1)))*prior).ravel()
            d = np.sort(s/np.hypot(s,.2))[::-1]
            p=len(d)
            u=np.full(p,.8/np.sqrt(2*(p-1)))
            u[0]=.8/np.sqrt(2)
            for n in [32,256]:
                for rep in range(16):
                    x = rng.normal(size=(n,p))
                    t = x@u+.6*rng.normal(size=n)
                    g = rng.normal(size=(64,p))*d
                    truth = g@u
                    y = truth + np.sqrt(1-np.sum((d*u)**2))*rng.normal(size=len(g))
                    order = rng.permutation(p)
                    for policy in POLICIES:
                        idx=allocate(d,n,policy,order)
                        pred,half,mod2=predict_axes(idx,x[np.arange(n),idx],t,g)
                        rows.append(dict(kernel=index,budget=n,rep=rep,policy=policy,
                                         mean_estimation_rmse=np.sqrt(np.mean((truth-pred)**2)),
                                         **summary(y,pred,half,mod2)))
            print("camera",index,flush=True)
    return rows


def scenes(paths,rng):
    rows = []
    for name,path in paths.items():
        cube=loadmat(path)[name.lower()+"_corrected"]
        h,w,_=cube.shape
        field=cube[:,:,:144].reshape(h,w,24,6).mean(axis=-1)
        target=cube[:,:,-16:].mean(axis=-1)
        strips=np.arange(h)*4//h
        for fold in range(4):
            train=field[strips==fold].reshape(-1,24)
            mu=train.mean(axis=0)
            eig,V=np.linalg.eigh(np.cov(train.T))
            ix=np.argsort(eig)[::-1]
            scale,V=np.sqrt(eig[ix]),V[:,ix]
            # Fix eigenvector signs across LAPACK builds.
            V *= np.sign(V[np.argmax(abs(V),axis=0),np.arange(24)])
            noise=.2*np.sqrt(eig.mean())
            d=scale/np.hypot(scale,noise)
            target_sample=target[strips==(fold+1)%4]
            tmean,tsd=target_sample.mean(),target_sample.std(ddof=1)
            cal=field[strips==(fold+2)%4].reshape(-1,24)
            xc=((cal-mu)@V)/scale
            tc=(target[strips==(fold+2)%4].ravel()-tmean)/tsd
            test_rows=np.flatnonzero(strips==(fold+3)%4)[::4]
            test=field[test_rows,::4].reshape(-1,24)
            y=(target[test_rows,::4].ravel()-tmean)/tsd
            raw_vectors=V*scale
            raw_vectors /= np.linalg.norm(raw_vectors,axis=1,keepdims=True)
            for n in [32,256]:
                for rep in range(16):
                    sample=rng.choice(len(cal),n,replace=False)
                    latent=xc[sample]
                    observed=(test-mu)@V+noise*rng.normal(size=test.shape)
                    g=observed/np.hypot(scale,noise)*d
                    order=rng.permutation(24)
                    for policy in (*POLICIES,"physical"):
                        if policy=="physical":
                            vectors=raw_vectors[np.resize(order,n)]
                            scores=np.sum(latent*vectors,axis=1)
                            pred,half,mod2=predict_scores(scores[:,None]*vectors,tc[sample],g)
                        else:
                            idx=allocate(d,n,policy,order)
                            pred,half,mod2=predict_axes(idx,latent[np.arange(n),idx],tc[sample],g)
                        rows.append(dict(scene=name,fold=fold,budget=n,rep=rep,policy=policy,
                                         target_scale=tsd,**summary(y,pred,half,mod2)))
            print(name,fold,flush=True)
    return rows


def run(out,paths):
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.SeedSequence(SEED)
    streams=[np.random.default_rng(s) for s in rng.spawn(3)]
    save(out,"tail_comparison",tails())
    save(out,"dimension",limits(streams[0]))
    save(out,"camera",camera(streams[1]))
    save(out,"scenes",scenes(paths,streams[2]))
    files=["finite_protocol.md","finite_design.py","finite_study.py","test_finite.py",
           "study.py","intervals.py","requirements.txt","replay.py"]
    record=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                platform=platform.platform(),seed=SEED,pid=os.getpid(),
                command="python finite_study.py",data={k:v[1] for k,v in SCENES.items()},
                kernel_archive=study.HASH,source={k:study.digest(k) for k in files},
                results={f.name:study.digest(f) for f in sorted(out.glob("*.csv"))})
    (out/"run.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",action="store_true")
    args=parser.parse_args()
    paths=inputs()
    if args.replay:
        reference=json.loads((REF/"run.json").read_text())
        for name,h in reference['source'].items():
            assert study.digest(name)==h, name
        out=Path(tempfile.mkdtemp(prefix="finite-replay-",dir="incoming"))
        run(out,paths)
        for name,h in reference["results"].items():
            assert study.digest(REF/name)==h, name
            compare(REF/name,out/name)
            print(name, "byte-identical" if study.digest(out/name)==h else "within tolerance")
    else:
        run(REF,paths)
