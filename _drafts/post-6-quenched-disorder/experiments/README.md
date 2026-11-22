# Computed figures for Disordered Minds

The author approved experiments 1–3 on November 21, 2026. These local CPU calculations accompany sections III–V of the working post. They answer different questions: what a restricted Bayesian model can learn, how a fixed-probe output law approaches a Gaussian process, and how a finite-step replica ansatz changes with temperature. The third is a numerical theory calculation, rather than a finite-network sampling experiment.

All commands run from the repository root. The environment uses Python 3.11 with the installed NumPy, SciPy, Matplotlib, and, for the replica solver, PyTorch. Each result directory records exact versions and source hashes. The environment variables select a writable Matplotlib cache and limit numerical-library threading.

```sh
MPLCONFIGDIR=/private/tmp/disordered-minds-mplconfig OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 _drafts/post-6-quenched-disorder/experiments/inductive_bias.py
MPLCONFIGDIR=/private/tmp/disordered-minds-mplconfig OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 _drafts/post-6-quenched-disorder/experiments/nngp_correspondence.py
```

The bias script accepts `--plot-only`; the NNGP script accepts `--compute` and `--plot` separately. They write numerical results beside themselves and light/dark PNGs plus a light PDF to `assets/disordered-minds/`. `plot_style.py` supplies the shared Alegreya typography and warm palette. Exact source snapshots accompany the saved results; the git HEAD is only the base revision, because this work is uncommitted.

## 1. Architectural symmetry and Bayesian learning

`inductive_bias.py` freezes 512 tanh features on circle inputs and infers only their Gaussian readout coefficients. It compares zero hidden bias with Gaussian hidden bias variance 1/4, reducing the hidden weight variance from 1 to 3/4 to preserve the marginal preactivation law. Dividing features by the square root of `512 * E[tanh(Z)^2]`, for standard normal Z, gives unit output variance after averaging over hidden draws. Empirical variance can differ for a finite feature realization.

The fixed target is `cos(2t) + 0.3 sin(t)`, with twelve noisy observations arranged in six antipodal pairs and observation standard deviation 0.05. Data seed 20261121 and paired feature seeds 100–107 are saved. The posterior panels use the predetermined seed 100. Hidden features stay fixed during conditioning; this does not implement inference over every network weight.

Without biases, odd tanh enforces `f(t+pi) = -f(t)` for every readout. All even Fourier modes, including the constant, are excluded. The target's even component has uniform-circle mean square 1/2, so the root mean squared error cannot fall below `sqrt(1/2)` in that architecture. Across eight feature draws, grid RMSE is **0.707707** without biases and **0.058396** with biases. The biased range is 0.053102–0.063173. The plotted upper-panel markers and shading are means and min–max ranges over those eight draws; the lower bands are two posterior standard deviations of the latent function.

This deliberately mismatched target demonstrates representational restrictions and their effect on inference. Its posterior-band inclusion fractions are descriptive, not a Bayesian calibration test. The finite hidden features add fixed structure alongside the quenched training dataset.

Validation compares kernel conditioning with direct inference over readout weights on a tiny problem: maximum mean error 4.33e-15 and variance error 2.44e-15. Independent adaptive quadrature checks the tanh normalization to 2.22e-15. PSD, antipodal symmetry, nonnegative posterior variance, and Fourier Parseval checks pass. Compute time after imports was about 0.13 seconds.

Saved evidence: `results/inductive-bias/results.json`, `results.npz`, `source_snapshot.py`, and `plot_style_snapshot.py`. The NPZ contains the observations, feature draws, prior spectra, and posterior predictions.

## 2. A finite network approaching its NNGP

`nngp_correspondence.py` fixes eight separated, non-antipodal circle probes, with geometry seed 2026112102. The geometric acceptance rule is recorded and applied before drawing outputs. Every network realization redraws all weights in a single hidden ReLU layer: hidden covariance I₂, readout variance 2, and output scaling 1/sqrt(N). Widths are 1, 2, 4, 8, 32, 128, and 512. The three smallest widths resolve the change before sampling noise dominates.

For unit-circle inputs with inner product rho, the analytic covariance is

\[
K(\rho)=\frac{\sqrt{1-\rho^2}+(\pi-\arccos\rho)\rho}{\pi}.
\]

It equals the unconditional covariance at **every** width, so covariance error would not measure convergence of the output law here. The site now animates the paired-output scatter as width increases, following the author's revision to remove the MMD panel. The static light/dark PNGs and PDF show only the width-2 and width-512 scatter comparison.

The animation uses 2,000 new independent sequences of hidden/readout weights, with seed 7701024. Each width-N network uses the first N neurons of its sequence and divides the sum by sqrt(N). The 29 saved widths are 1–16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, and 1024. Points retain identity across frames; connecting them does not depict training or imply that an individual path settles to one GP draw. The empirical distribution approaches the fixed Gaussian law.

The two probes are indices 0 and 2 of the same original fixed geometry. Their limiting covariance has diagonal 1 and off-diagonal 0.4890421496. The JSON includes analytic Gaussian 95% ellipse coordinates and reports out-of-view counts on fixed axes [-4.5,4.5]. Those counts are 26, 16, 0, and 0 at widths 1, 2, 512, and 1024. These draws differ from the original static scatter draws, whose width-2 count is 14. `assets/disordered-minds/nngp-width-animation.json` is about 1.08 MB; full-precision points and model/configuration are saved as `animation_data.npz` and `animation_config.json` under `results/nngp/`. Independent explicit prefix sums agree to 1.04e-17, covariance agrees within 2.49 estimated standard errors, and rounding to five decimal places changes exported coordinates by less than 5e-6.

The following joint-distribution calculation is retained as supplemental numerical evidence, rather than shown in the site figure. All eight outputs are whitened with the fixed analytic K and compared with Gaussian draws using unbiased squared RBF maximum mean discrepancy. The squared bandwidth is 16. This choice is fixed across widths and is not tuned to the observed outputs.

Each reported value averages eight independent comparisons, with 1,024 draws per distribution in each batch; the table gives two empirical standard errors across those batches. The matched null consists of 32 independent groups of eight equally sized Gaussian-versus-Gaussian comparisons. Its central 90% range is **[-0.000192046, 0.000219516]**, and its mean is -0.457 standard errors from zero. All exact seeds are in `config.json`.

| Width | Mean estimated MMD² | Two standard errors |
|---|---:|---:|
| 1 | 0.00541768 | 0.00098613 |
| 2 | 0.00211538 | 0.00037497 |
| 4 | 0.00061460 | 0.00017854 |
| 8 | 0.00040612 | 0.00024515 |
| 32 | -0.00011900 | 0.00014611 |
| 128 | -0.00003468 | 0.00014044 |
| 512 | -0.00005020 | 0.00018654 |

The discrepancy decreases into the reference sampling range. At widths 32–512, this experiment cannot resolve the remaining difference from the GP. Negative finite estimates are retained because the unbiased U-statistic is signed, even though population MMD² is nonnegative. The null quantiles describe an empirical sampling range, not a strict bound or a confidence interval for a particular network comparison.

The static scatter panels show 2,000 separate initializations at widths 2 and 512, using probes 0 and 2 chosen in advance. The dashed ellipse contains 95% of the limiting bivariate Gaussian. Both use the central [-4.5, 4.5] view; 14/2,000 and 0/2,000 realizations, respectively, lie outside, as labeled in the figure.

Independent angular quadrature checks the analytic kernel to maximum error 2.66e-9; its minimum eigenvalue is 0.00159068. Empirical covariance entries agree with the exact covariance within a maximum 3.30 estimated standard errors over the tested widths. A tiny direct double sum exactly matches the discrepancy implementation. The exact single-output excess kurtosis is 15/N; noisy sample estimates are retained but are not the displayed convergence test. The final compute pass took 6.69 seconds after imports.

Saved evidence: `results/nngp/{config.json,metrics.json,plot_metadata.json,data.npz,source/}`. The NPZ retains probes, kernel, metric batches, Gaussian references, and display draws. The fixed-probe experiment does not establish rare-input thermodynamics when dimension grows with width.

Primary background: [Zavatone-Veth's NNGP notes](https://jzv.io/assets/pdf/lecture_notes_on_nngp_from_mft.pdf), [Lee et al.](https://arxiv.org/abs/1711.00165), and [Matthews et al.](https://arxiv.org/abs/1804.11271).

## 3. A temperature-driven replica bifurcation

`replica_bifurcation.py` and `rsb_solver.py` independently implement the one-hidden-layer sine problem in [Winer and Hanin, *Deep Neural Nets as Hamiltonians*, v2](https://arxiv.org/html/2503.23982v2), especially sections 3.1, 3.4, 3.6 and Appendix A.2. Their Figure 2 motivates the overlap-versus-temperature observable. Its caption does not identify the aspect ratio, so this calculation explicitly sets alpha=N/d=1 and does not claim the published critical temperature.

The model uses Gaussian reference inputs, hidden weights with variance 1/d, Gaussian readout weights with variance one, and no 1/sqrt(N) output normalization. It studies the proportional d,N→∞ replica ansatz. It is not a finite-network Monte Carlo experiment, and no published data were digitized. The finite hierarchy has K=20 fixed breakpoints m_j=j/21 and 21 off-diagonal overlap levels. The direct functional minimizes that profile at fixed diagonal self-overlap D, then maximizes over D. The periodic sine integrals use a 256-point Fourier grid and 128-point Gaussian readout quadrature. Full equations, conventions and limitations are documented in `results/rsb/README.md`.

```sh
MPLCONFIGDIR=/private/tmp/disordered-minds-mplconfig OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 _drafts/post-6-quenched-disorder/experiments/replica_bifurcation.py run
MPLCONFIGDIR=/private/tmp/disordered-minds-mplconfig OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 _drafts/post-6-quenched-disorder/experiments/replica_bifurcation.py check
MPLCONFIGDIR=/private/tmp/disordered-minds-mplconfig OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 _drafts/post-6-quenched-disorder/experiments/replica_bifurcation.py plot
```

The sweep runs from beta=0.5 to 8, with extra resolution around branching. At beta=3.5 the overlap levels coincide to about 5e-6; by beta=3.75 they span 0.817985–0.858846. An independent RS local-curvature calculation changes sign from +0.033082 to -0.081893 over that interval. At beta=7.1, the K20 sweep endpoints are 0.950662 and 1.592967; its diagonal self-overlap is 2.168603. These overlaps divide by dimension, rather than by the input norms, so values above one are allowed.

The largest saved saddle residual is below 6e-6. A distinct broad initial profile at beta=7.1 reaches the same functional within 1.3e-11, with endpoint differences below 1.5e-4. The exact linear functional and its saddle gradients agree to machine precision; direct scalar adaptive integration checks the terminal nonlinear convolution, and central finite differences check automatic derivatives. At every reported sweep solution, the Fourier convolution's numerical floor is inactive. Finer field and readout quadrature checks are retained in `refinements.json` alongside hierarchy checks.

This demonstrates a branching numerical solution within a finite-step replica ansatz and agreement with loss of local RS stability. It does not prove a global equilibrium variational principle or convergence to full RSB. No tanh control was run; the exact linear case is the activation control. Source snapshots, metadata, all overlap levels, residuals, checks and logs are retained under `results/rsb/`.
