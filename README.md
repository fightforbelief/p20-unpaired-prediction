# Sharp conditional prediction from unpaired Gaussian measurements

Yunjie Guo and Hanqing Yang · Independent Researcher, Suzhou, China

[Read the manuscript](paper/manuscript.pdf) · [LaTeX source](paper/manuscript.tex) · [Numerical design](protocol.md)

A sensor observes a noisy linear image of a visible field. A separate archive contains measurements of a scalar target in a missing channel. Knowing the two Gaussian marginals leaves their cross-covariance undetermined. This paper computes the shortest interval for a given observation that covers the target across every compatible jointly Gaussian population.

The interval has a numerical upper bound and an explicit coupling witnessing the lower bound. Theorem 3 identifies the optimal fixed subspace to measure jointly with the target. Theorem 4 shows how to choose a single direction for a current query and gives matching upper and lower orders for the number of paired calibration objects needed.

## Reproduce

The recorded run used Windows, Python 3.12.10, NumPy 2.2.6 and SciPy 1.15.3. From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements.txt
& $py study.py --audit
& $py -m unittest -v
& $py replay.py
```

The audit downloads the [eight Levin camera-shake kernels](https://cabouman.github.io/grad_labs/labs/resources/levin09_kernels.zip) on first use and checks the SHA-256 digest. For an offline run, place the identical archive in `incoming/levin09.zip`. See the [original paper](https://people.csail.mit.edu/billf/papers/deconvLevinEtalCVPR09.pdf) and the [distribution notes](https://cabouman.github.io/grad_labs/labs-60141/lab3/index.html).

The replay writes to a fresh folder under `incoming/`, checks every output column against the reference files and preserves the original run. It allows small floating-point differences across numerical libraries. To regenerate the canonical CSVs and machine record directly, run `$py study.py` with the PowerShell call operator `&`. Build the PDF with `tectonic manuscript.tex` from `paper/`.

## Using the solver

```python
from intervals import sharp_radius

r = sharp_radius(gains=[0.90, 0.35], observation=[2.0, -0.7], alpha=0.05)
interval = (-r.upper, r.upper)
least_favorable_candidate = r.coupling
```

Supply a standardized target, the singular-coordinate gains of the whitened forward map, and a measurement in standardized left singular coordinates. The returned interval is in units of the target marginal standard deviation. Its lower half-length is attained at the returned coupling. The gap is controlled by `tol` up to floating-point accuracy.

## Evidence and interpretation

- [Operator spectra](results/operators.csv) give trace and paired-design results in 48 settings.
- [Conditional intervals](results/conditional.csv) compare sharp and equal-tail union widths on 128 queries, including the conditional law at the feasible witness.
- [Paired calibration](results/paired.csv) reports 50,000 simulations per sample-size and coupling cell.
- [Run record](results/run.json) contains the input, code and output digests, package versions and random seed.

The kernels are measured. Visible-field spectra, queries and calibration responses are model-based. The conditional guarantee assumes joint Gaussianity, known marginal parameters and known noise. The finite-calibration result conditions on the query and on calibration scores and averages over calibration responses. Testing a learned natural-image prior would require separate paired validation data.
