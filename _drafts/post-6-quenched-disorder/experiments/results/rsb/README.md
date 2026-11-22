# A replica saddle that separates into several overlap levels

This experiment numerically solves a finite-step Parisi ansatz for a one-hidden-layer sine network. It follows the model and hierarchical integrals of Mike Winer and Boris Hanin, [*Deep Neural Nets as Hamiltonians*, arXiv:2503.23982v2](https://arxiv.org/html/2503.23982v2). It is an independent implementation of their one-layer problem, with an explicitly specified aspect ratio. We did not use author code, digitize their figure, or infer replica symmetry breaking from finite-network MCMC.

The plotted sweep uses 20 hierarchy breakpoints (21 overlap levels), 256 Fourier points over one period, and 128-point Gauss–Hermite readout quadrature. Its 20 inverse temperatures run from $\beta=0.5$ to $8$. The hierarchy stays collapsed through the sampled $\beta=3.5$ point, and has a resolved width $q_{\max}-q_{\min}=0.04086$ at $\beta=3.75$. Independently, the restricted RS saddle's normalized replicon changes from $+0.03308$ to $-0.08189$ across the same interval, supporting the interpretation as local RS instability rather than an arbitrary spread introduced by initialization. This brackets the transition on our grid; it does not locate a critical temperature precisely.

At $\beta=7.1$, the plotted 20-step saddle has $q_{\min}=0.95066$, $q_{\max}=1.59297$, and $D=2.16860$. The largest absolute diagonal or projected profile residual over the sweep is $5.93\times10^{-6}$, and every saved optimizer reports success. A second, much broader initial profile at this temperature returns the same functional within $6\times10^{-12}$ and endpoints within $1.5\times10^{-4}$. Two starts do not establish global uniqueness.

| Refinement at $\beta=7.1$ | Largest endpoint change from the plotted sweep | Change in $\mathbb E\log Z/d$ functional |
| --- | ---: | ---: |
| 256 to 512 Fourier points, fixed 20 steps and readout quadrature | $4.9\times10^{-5}$ | $1.4\times10^{-11}$ |
| 128 to 256 readout quadrature points, fixed 20 steps and Fourier grid | $1.6\times10^{-4}$ | $9.4\times10^{-8}$ |
| 20 to 50 hierarchy steps, fixed spatial/readout grids | $6.9\times10^{-3}$ | $3.25\times10^{-6}$ |

At $\beta=4.5$, the 20-to-50-step endpoint shift is below $4.6\times10^{-4}$. The hierarchy resolution is the leading numerical limitation at the colder check point; the main figure deliberately presents the actual 20-step levels rather than implying that their endpoints are converged to many decimal places. Every refinement succeeds with a saddle residual below $6.3\times10^{-7}$.

The independent checks include an exact finite-integer block-matrix determinant, an adaptive Gaussian integral for the terminal nonlinear site, central finite differences of the functional, and the exact linear-network solution. The determinant error is $5.6\times10^{-17}$, terminal integral error is below $1.8\times10^{-15}$ at the three checked configurations, and the largest tested derivative error is $1.7\times10^{-10}$. For a linear activation, the functional and all hierarchy derivatives match $c=1$, $q_i=\alpha\beta^2$, $D=1+\alpha\beta^2$, and $\mathbb E\log Z/d=\alpha\beta^2/2$ to floating-point precision at $\beta=0,1,4.5,7.1$ (§4 of the paper).

The FFT positivity floor is inactive at every reported sweep and refinement saddle: zero clipped grid points and zero affected readout quadrature mass. The smallest raw convolution argument over the main sweep is $2.44\times10^{-5}$, comfortably above its $10^{-15}$ numerical floor. These diagnostics apply to the reported saddles, not arbitrary intermediate optimizer trials.

The network is

\[
H_W(x)=\sum_{i=1}^{N} a_i\sin(w_i^\top x),\qquad
w_{ij}\sim\mathcal N(0,1/d),\quad a_i\sim\mathcal N(0,1),
\]

and the input Gibbs measure is proportional to

\[
\exp\{-\|x\|^2/2-\beta H_W(x)\}\,dx.
\]

The readout has variance one, as in the paper, rather than the usual NNGP variance $1/N$. We take $N,d\to\infty$ with $\alpha=N/d=1$. A plotted overlap is $q=x^\top x'/d$, not cosine similarity. The diagonal self-overlap is $D=\|x\|^2/d$ and is optimized along with the off-diagonal profile. These conventions follow paper §3.1, equations (3)–(6), and §3.4, equations (23)–(24). Figure 2 and §6.3 motivate the observable; its caption does not specify an aspect ratio, so our numerical transition should not be equated with the caption's approximate $\beta\simeq4.5$.

For one hidden layer, the replicated Gaussian input measure gives an entropic term $[\log\det Q-\operatorname{tr}Q+n]/2$ per input dimension. The final readout weights can be integrated exactly, giving a hidden-site factor

\[
\mathcal Z(Q)=\mathbb E_{a\sim\mathcal N(0,1)}
\mathbb E_{z\sim\mathcal N(0,Q)}
\exp\left(\beta a\sum_b\sin z_b\right).
\]

The sign of $\beta a$ is immaterial because the Gaussian readout is symmetric. We evaluate the $n\to0$ coefficient of $\log\mathcal Z$ using the hierarchical Gaussian recursion in Appendix A.2, equations (44)–(49). Because there is only one hidden layer, the readout field $a$ is common to all hierarchy levels; its Gaussian average is taken *after* the logarithmic recursion. No negative-variance dual-field integral is needed.

Write the off-diagonal hierarchy as $q_0\leq\cdots\leq q_K$ with fixed breakpoints $m_j=j/(K+1)$, and let $c=D-q_K$. With $\chi_i=c+\sum_{j>i}m_j(q_j-q_{j-1})$, the continued input entropy is

\[
\lim_{n\to0}\frac{\log\det Q}{n}
=\log c+\frac{q_0}{\chi_0}
+\sum_{j=1}^{K}\frac{1}{m_j}\log\frac{\chi_{j-1}}{\chi_j}.
\]

The functional is half of this expression minus $(D-1)/2$, plus $\alpha$ times the hierarchical site contribution. The off-diagonal profile is minimized at fixed $D$, and the resulting functional is maximized over $D$. An unconstrained minimization of both would solve a different problem. The implementation uses SLSQP with analytic automatic derivatives for the inner profile and a bounded scalar search for the outer self-overlap.

Every sine-site function is $2\pi$ periodic, so Gaussian convolution is the Fourier multiplier $\exp(-v k^2/2)$ for variance $v$. This retains the full scalar nonlinear integral rather than approximating the network as an NNGP. Gaussian readout integration uses Gauss–Hermite quadrature. Small nonzero hierarchy increments seed each collapsed continuation point, because an exactly replica-symmetric initialization can remain symmetric even when an RSB solution exists (§3.6).

`rsb_solver.py` contains the functional and the saddle solver. `replica_bifurcation.py` contains the sweep, independent numerical checks, refinements, and plots. `sweep.json` saves every overlap level, hierarchy weight, diagonal, objective, residual, status, runtime and software/source fingerprint. `second-start.json` retains the distinct-start control. `checks.json` and `refinements.json` retain validation rather than hiding it behind the figure. Exact commands, from the repository root, are

```sh
python3 _drafts/post-6-quenched-disorder/experiments/replica_bifurcation.py run
python3 _drafts/post-6-quenched-disorder/experiments/replica_bifurcation.py check
python3 _drafts/post-6-quenched-disorder/experiments/replica_bifurcation.py plot
```

The `sources/` directory retains byte-identical source snapshots for both stages: files ending `_sweep.py` match every hash in `sweep.json` and `second-start.json`, while files ending `_validation.py` match `checks.json` and `refinements.json`. The later stage adds final-point diagnostics and a narrower optional diagonal bracket for refinement, without changing the functional. The common plotting helper is also retained. A hash audit confirmed that every saved source fingerprint has a matching snapshot.

This is numerical evidence within a finite-step hierarchical replica ansatz. Finite-step convergence, saddle convergence, and the exact linear-network limit check the implementation; they do not prove that this ansatz gives the global thermodynamic free energy or establish full RSB as a mathematical theorem. There is no sampling claim about a finite trained network. The only control activation solved here is the exact linear case; no unrun tanh comparison is implied.
