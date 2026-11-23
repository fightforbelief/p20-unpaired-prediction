"""Typeset the verified CSV summaries. Optional dependency: Matplotlib."""
import csv
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/"paper"
DATA = ROOT/"results/finite"
NAMES = {"greedy":"Exact allocation","mean_info":"Mean information","proportional":"Proportional",
         "balanced":"Balanced","top4":"Leading four","physical":"Band groups"}


def load(name):
    with (DATA/f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


def select(rows, **query):
    return [r for r in rows if all(r[k]==str(v) for k,v in query.items())]


def average(rows, key):
    return mean(float(r[key]) for r in rows)


def camera_table():
    c = load("camera")
    rows = []
    for policy,label in NAMES.items():
        if policy=="physical":
            continue
        cells = [label]
        for n in [32,256]:
            group=select(c,budget=n,policy=policy)
            cells.extend(f"{average(group,k):.2f}" for k in ["mean_modulus2","half50","mean_estimation_rmse"])
        rows.append(" & ".join(cells)+r"\\")
    (HERE/"camera_rows.tex").write_text("\n".join(rows)+"\n")


def scene_table():
    data=load("scenes")
    rows=[]
    for n in [32,256]:
        for policy,label in NAMES.items():
            cells=[str(n),label]
            for scene in ["Indian_pines","Salinas"]:
                group=select(data,scene=scene,budget=n,policy=policy)
                cells.extend([f"{average(group,'half50'):.2f}",f"{100*average(group,'coverage'):.1f}",
                              f"{average(group,'rmse'):.2f}"])
            rows.append(" & ".join(cells)+r"\\")
        if n==32:
            rows.append(r"\midrule")
    (HERE/"scene_rows.tex").write_text("\n".join(rows)+"\n")


def figure():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(6.65,2.85),layout="constrained")
    colors=["#245586","#ad5024","#31796b","#675482"]
    dim=load("dimension")
    ps=[8,32,128]
    for color,mode,power,label,marker in [
        (colors[0],"reusable",1,"Reusable, $N=p$","o"),
        (colors[1],"query",1,"One query, $N=p$","s"),
        (colors[2],"reusable",2,"Reusable, $N=p^2$","^")]:
        ys=[average(select(dim,p=p,budget=p**power,mode=mode),"root_mean_modulus2") for p in ps]
        ax.plot(ps,ys,label=label,c=color,marker=marker,ms=4,ls="--" if power==2 else "-")
    ax.set(xscale="log",xlabel="Visible dimensions, $p$",ylabel=r"Root mean squared modulus")
    ax.set_xticks(ps,ps); ax.minorticks_off()
    ax.legend(frameon=False,loc="upper left",fontsize=7)
    ax.set_title("A   Budget scaling",loc="left",fontweight="bold",fontsize=9)
    batch=load("batches"); sizes=[1,4,16,64]
    for color,policy,label,marker in [(colors[0],"reusable","Reusable","o"),(colors[1],"batch","Batch spectrum","s"),
                                       (colors[2],"first_query","First query","^")]:
        ys=[];err=[]
        for m in sizes:
            vals=[float(r["rms_half"]) for r in select(batch,budget=256,policy=policy,batch=m)]
            ys.append(mean(vals));err.append(2*stdev(vals)/math.sqrt(len(vals)))
        bx.errorbar(sizes,ys,yerr=err,label=label,c=color,marker=marker,ms=4,capsize=2)
    bx.set(xscale="log",xlabel="Queries sharing 256 pairs",ylabel="Mean batch RMS half-width")
    bx.set_xticks(sizes,sizes); bx.minorticks_off()
    bx.legend(frameon=False,fontsize=7,loc="lower right")
    bx.set_title("B   Shared calibration",loc="left",fontweight="bold",fontsize=9)
    for a in (ax,bx):
        a.grid(axis="y",alpha=.18); a.set_axisbelow(True)
    fig.savefig(HERE/"allocation.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__=="__main__":
    figure()
    camera_table()
    scene_table()
