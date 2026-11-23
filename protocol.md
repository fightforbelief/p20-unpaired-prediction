# Numerical design

The mathematical claims concern all centered Gaussian completions of two unpaired marginals. The numerical study evaluates the sharp conditional radius on measured forward operators. It uses simulated observations and targets.

Use all eight Levin camera-shake kernels on a 32 by 32 periodic grid. Use flat input variance and a smooth input spectrum with variance 1/(1+64 times the sum of the squared sines of the two angular half-frequencies). Noise standard deviations are 0.05, 0.2 and 0.8. Compute the trace of squared gains and its residual after 32 top or uniformly random orthogonal paired directions for every setting. The middle noise and smooth spectrum are the primary setting. Sort gains and share 16 independent standard-normal test vectors across kernels. Compute the radius to absolute tolerance 0.0005 and compare with the full equal-tail union. Check coverage of the returned witness.

For kernel 1 and test vector 1, calculate coverage and median half-length over 50,000 independent calibration/target replicates per cell. Acquire n=4,16,64,256 paired scores along the realized conditional mean of the visible field. Use latent couplings of length 0, 0.5 and 0.95 in this direction. Include an interval that ignores calibration error. Use PCG64 seed 20261123 throughout. These settings illustrate scaling. They do not estimate natural-image coverage.
