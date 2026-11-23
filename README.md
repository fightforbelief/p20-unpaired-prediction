# How many paired measurements does conditional prediction require?

Yunjie Guo and Hanqing Yang · Independent Researcher, Suzhou, China

[Manuscript](paper/manuscript.pdf) · [LaTeX](paper/manuscript.tex) · [Finite-archive study protocol](finite_protocol.md) · [Novelty and strength assessment (中文)](review/revision.md) · [Earlier strength audit](review/review.md)

A visible sensor field and a missing scalar target have known Gaussian marginal laws and unknown cross-covariance. Each additional calibration object supplies its target and one scalar projection of its field. How should an archive spend a fixed budget, and what changes if the future queries are already known?

- For realized calibration readings and a query, the pointwise minimax expected interval length is of order one plus the square root of a computable quadratic form. Its matrix accounts for the finite scores, their directions and their known measurement reliabilities.
- For an archive built before the queries arrive, the optimal order over arbitrary fixed linear readings is characterized by a spectral allocation problem. A discrete allocation minimizes expected squared information radius within the eigenaxis catalogue and attains the unrestricted order at unit reliability.
- In an isotropic field with dimension p, bounded average pointwise minimax length costs order p squared paired objects for a reusable archive and order p for an archive acquired for one query. For a known batch, the singular spectrum of its posterior mean vectors determines the cost of sharing pairs. The guarantee is for each query individually.
- Without pairs, the shortest robust Gaussian interval has an exact scalar optimization formula. A single-support approximation is effectively exact on the original 128 camera queries. Intermediate-signal comparisons report where the remaining gap matters.

## Reproduce

The recorded run used Windows, Python 3.12.10, NumPy 2.2.6 and SciPy 1.15.3. From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements.txt
& $py study.py --audit
& $py -m unittest -v
& $py replay.py
& $py finite_study.py --replay
& $py span_check.py --replay
```

The audit downloads the [eight Levin camera-shake kernels](https://cabouman.github.io/grad_labs/labs/resources/levin09_kernels.zip) on first use and checks the SHA-256 digest. For an offline run, place the identical archive in `incoming/levin09.zip`. See the [original paper](https://people.csail.mit.edu/billf/papers/deconvLevinEtalCVPR09.pdf) and the [distribution notes](https://cabouman.github.io/grad_labs/labs-60141/lab3/index.html).

The replay writes to a fresh folder under `incoming/`, checks every output column against the reference files and preserves the original run. It allows small floating-point differences across numerical libraries. To regenerate the canonical CSVs and machine record directly, run `$py study.py` with the PowerShell call operator `&`. Build the PDF with `tectonic manuscript.tex` from `paper/`.

The finite study also uses the corrected AVIRIS Indian Pines and Salinas cubes. The script downloads checksum-pinned files on first use. Alternatively copy them to `incoming/p20-Indian_pines_corrected.mat` and `incoming/p20-Salinas_corrected.mat`. The raw images are hosted by their distributors. The [scene documentation](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) describes the measurements. Run `& $py finite_study.py` to regenerate the reference finite-archive outputs.

Run `python paper/figures.py` from the project root to regenerate the figure and numeric table fragments from the archived CSVs (Matplotlib is an optional plotting dependency). The prebuilt manuscript can be compiled from its LaTeX sources and included figure with Tectonic alone.

## Using the solver

```python
from intervals import sharp_radius

r = sharp_radius(gains=[0.90, 0.35], observation=[2.0, -0.7], alpha=0.05)
interval = (-r.upper, r.upper)
least_favorable_candidate = r.coupling
```

Supply a standardized target, the singular-coordinate gains of the whitened forward map, and a measurement in standardized left singular coordinates. The returned interval is in units of the target marginal standard deviation. Its lower half-length is attained at the returned coupling. The gap is controlled by `tol` up to floating-point accuracy.

## Evidence and interpretation

In a 64-dimensional Gaussian experiment with 16 queries and 256 pairs shared across the batch, tailoring the directions reduced mean batch RMS half-width from 8.25 to 4.39 target standard deviations (paired Monte Carlo standard error of the difference 0.06). The simpler mean-information rule achieved 4.36. With 256 pairs on measured spectra, empirical coverage of the exact allocation was 97.0% on Indian Pines and 56.8% on Salinas. The lowest Salinas strip covered 25.5% at the nominal 95% level.

- [Operator spectra](results/operators.csv) give trace and paired-design results in 48 settings.
- [Conditional intervals](results/conditional.csv) compare sharp and equal-tail union widths on 128 queries, including the conditional law at the feasible witness.
- [Paired calibration](results/paired.csv) reports 50,000 simulations per sample-size and coupling cell.
- [Run record](results/run.json) contains the input, code and output digests, package versions and random seed.
- [Finite archive results](results/finite/) report camera and AVIRIS comparisons by archive, conditional-modulus scaling, shared-query experiments, tail-allocation checks and source hashes.
- [Within-span control](results/span/comparison.csv) isolates the gain from allocating differently inside the same known query span. Its [protocol](span_protocol.md) records its supplemental status.
- [Target-marginal shift by held-out strip](results/scene_shift.csv) records a descriptive check added after inspecting coverage. Run `& $py scene_shift.py` to recompute it.
- [Prior audit data](review/baseline-audit.csv) accompany the baseline proof. The review concerns the initial commit 42aa864.

The Gaussian theory assumes known marginals and independent Gaussian instrument noise. Finite-archive coverage conditions on the visible query and all calibration scores and averages over calibration responses. The camera study uses measured operators with simulated responses. The AVIRIS study predicts measured hidden bands in contiguous held-out spatial strips, using estimated marginal moments and injected deployment noise. It measures transfer to those scenes. Spatial shift and non-Gaussianity can invalidate the nominal coverage. Each dense spectral projection is charged as one instrument reading in the theory. Physical implementation has to provide that measurement at the assumed cost.
