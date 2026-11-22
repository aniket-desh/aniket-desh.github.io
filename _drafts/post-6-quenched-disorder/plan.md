# Post 6 planning draft

Saved November 21, 2026. This file retains the planning and citations behind the working site draft, [Disordered Minds](../../blog/disordered-minds/index.html), using the author's revised introduction and the continuation from [opening.md](opening.md). The author subsequently approved the full draft and experiments 1–3; implementation and measured results are recorded below and in [experiments/README.md](experiments/README.md).

The organizing question follows the reader from the forge to neural networks. What stays fixed while something else explores its possible states? The metallurgy analogy gives that question a physical meaning, and the replica method gives us a way to calculate its consequences.

The intended reader is a technical person encountering this physics from outside the field. Keep the metallurgy opening and the later physical introduction to quenched disorder and replicas brief. Most of the post should concern neural networks. Introduce physical terms in plain language and use a small spin visual for frustration. Figures should use the site's serif typography and warm brown, ochre, and terracotta colors drawn from the Vulcan painting.

The derivations in planning items 2–3 belong to one compact statistical-physics section after the opening. Use the random-field calculation as its worked example, with longer Wishart calculations and mathematical qualifications available in optional notes. Items 4–7 supply the main neural-network discussion.

The taxonomy needs a small refinement. Frozen training inputs already belong to a frozen dataset, and both Winer–Hanin and the ARC preprint hold network weights fixed while averaging over inputs. Their observables differ. Organize the post around three perspectives, while making those relationships explicit.

| Perspective | What stays fixed | What varies | What it explains |
|---|---|---|---|
| Learning from a dataset | Training examples, labels, and any teacher realization | Parameters drawn from a posterior or Gibbs measure | Generalization, capacity, uncertainty, and the geometry of solutions |
| Exploring a particular network | A realization of the weights | Inputs drawn from a specified distribution, possibly tilted toward extreme outputs | Typical behavior, rare events, and input-landscape geometry |
| Studying random initialization | A collection of probe inputs | Randomly initialized weights | Output distributions, kernels, and the Gaussian-process limit |

Winer–Hanin belongs in the second row. ARC also belongs there, estimating an ordinary Gaussian-input expectation for a particular network, with estimation error subsequently averaged over random weights. Jacob's notes span several rows. His perceptron notes concern storage constraints, his NNGP notes average over weights, and his Wishart notes provide an accessible replica calculation. [Winer–Hanin](https://arxiv.org/html/2503.23982v2), [ARC preprint, revised November 2, 2026](https://arxiv.org/abs/2605.05179v3), [Jacob's notes](https://jzv.io/notes).

## Approved completion and implementation

Approved November 21, 2026. The working site draft includes the continuation through section VII, with narrative section headers modeled on the author's other posts. The approved opening is retained, and extended calculations and experimental methods live in six native expandable appendix entries. Computed figures now accompany the Bayesian, frozen-weight landscape, and NNGP sections. Network-mode diagrams remain explicitly schematic.

The current section titles are I. *Frozen in the Forge*, II. *Seeing Double*, III. *Lessons in the Grain*, IV. *Walking on Frozen Ground*, V. *A Different Cast*, VI. *The Average Within*, and VII. *What the Fire Leaves Behind*. The appendix is *Beneath the Surface*.

The organizing structure is three ensembles defined by what is held fixed and what is averaged. Winer–Hanin and ARC both condition on a network's weights and average over inputs. The NNGP perspective conditions on probe inputs and averages over initializations. An ordinary weight average alone does not define a quenched free energy.

### I–III. Existing opening and Bayesian learning

Keep *Frozen in the Forge*, *Replicas*, and *Learning from Frozen Data* as approved. They establish the analogy, the statistical mechanics, and the first neural-network ensemble.

### IV. A Frozen Network, a Landscape of Inputs

- Hold a realization of the weights fixed and introduce the tilted input measure \(p_{\beta,W}(x)=p_0(x)e^{-\beta H_W(x)}/Z_W\). Explain how temperature selects different output regions and how the input measure contributes to the model.
- Solve a linear neuron under Gaussian inputs by completing the square, then introduce the input measure and extensive normalization used by Winer–Hanin. State the growing dimensions and their ratios alongside the calculation.
- Introduce input replicas and overlaps to describe the geometry of important inputs. Explain the paper's activation-dependent RS and RSB solutions, distinguishing analytic special cases from numerical solutions of replica equations. Keep the full Parisi calculation optional.
- Open with the network diagram in frozen-weights mode. Replace the overlap schematic with the approved numerical replica-branch calculation, stating its aspect ratio and finite-step scope.

Primary source: [Winer–Hanin, v2](https://arxiv.org/html/2503.23982v2).

### V. The Same Inputs, Many Networks

- Fix a finite collection of probe inputs and average over randomly initialized weights. Derive the Gaussian-process limit through a single hidden layer and then the covariance recursion through depth.
- State the fixed-depth, fixed-probe limit and connect the resulting prior over functions back to Bayesian conditioning in III. Explain how this differs from proportional limits with growing sample count or input dimension.
- Explain what this ensemble says about joint output statistics and what additional analysis is needed for input landscapes of a particular network. Tie RSB to the chosen configuration measure rather than to Gaussian-process terminology.
- Open with the same network diagram in fixed-probes mode, with changing prior weight samples.

Primary source: [Zavatone-Veth's NNGP notes](https://jzv.io/assets/pdf/lecture_notes_on_nngp_from_mft.pdf).

### VI. What One Network Does on Average

- Return to one fixed network and define \(m_W=\mathbb E_x f_W(x)\). Explain why averaging \(m_W\) across initializations can discard the instance-specific quantity we want to estimate.
- Use the identity \(\mathbb E_W m_W^2=\mathbb E_{W,x^1,x^2}[f_W(x^1)f_W(x^2)]\), with independent input draws, to connect the estimation question to two replicas sharing the same weights.
- Introduce ARC's cumulant propagation as a computational payoff. Explain how activation statistics are propagated through a known network and distinguish the polynomial-activation, fixed-depth theoretical guarantees from empirical results for other activations.
- Use a compact diagram comparing repeated input sampling with propagating distribution statistics through one fixed network. This section is a further application of the frozen-weight ensemble.

Primary source: [ARC preprint, v3](https://arxiv.org/html/2605.05179v3).

### VII. Choosing the Ensemble

- Bring the three perspectives together in one table covering fixed variables, fluctuating variables, observables, and large-system limits.
- State what RS and RSB describe in the corresponding ensembles, and distinguish a claim about a limiting overlap distribution from finite-size sampling behavior.
- Return briefly to the forge. The connection is how a fixed preparation constrains the configurations or responses we can explore, with each neural-network ensemble making that question precise.

Keep longer replica algebra and full algorithmic error bounds in optional notes. The main text should state the input measure, normalization, and limiting regime when each is introduced. Distinguish measured results in the approved experiments from the earlier proposals retained below. The intended completed post is roughly 4,500–5,500 words excluding references and optional notes, subject to the author's revisions.

## Background derivations and earlier planning

### 1. Begin in the forge, then identify the timescales

Use the smith cooling hot steel to introduce how a preparation process leaves persistent microscopic structure. In steel, sufficiently rapid cooling suppresses diffusional transformations and can produce martensite. That provides the bridge to statistical mechanics, where some variables remain effectively fixed throughout an observation while others continue fluctuating. The physical connection rests on kinetics and timescales. [Cambridge metallurgy practical](https://www.phase-trans.msm.cam.ac.uk/2000/practicals/AP3/AP3.html).

Write the separation explicitly,

\[
\tau_{\mathrm{fast}}\ll t_{\mathrm{observation}}\ll\tau_{\mathrm{disorder}}.
\]

Then move to magnetic impurities embedded in a metal. Their positions determine interactions between magnetic moments, while the moments can change orientation. This gives the reader a concrete route from metallurgy into disordered systems. Edwards's own account also connects replica calculations in permanently crosslinked rubber to their use in spin glasses, which could make a historical footnote. [Edwards's retrospective](https://garfield.library.upenn.edu/classics1985/A1985APZ5700001.pdf).

The proposed hero is Velázquez's *The Forge of Vulcan* (1630), depicting the Roman counterpart of Hephaestus. Its workshop setting supports the opening. [Metropolitan Museum discussion](https://resources.metmuseum.org/resources/metpublications/pdf/Juan_de_Pareja_by_Diego_Velazquez_The_Metropolitan_Museum_of_Art_Bulletin_v_29_no_10_Part_II_June_1971.pdf).

### 2. Make the distinction between quenched and annealed averages tangible

Introduce a configuration \(s\), frozen disorder \(J\), and

\[
Z_J(\beta)=\int\mu(ds)\,e^{-\beta H(s;J)}.
\]

Each realization has its own free energy. Averaging those free energies gives

\[
f_{\mathrm q}=-\frac{1}{\beta N}\mathbb E_J\log Z_J,
\qquad
f_{\mathrm a}=-\frac{1}{\beta N}\log\mathbb E_J Z_J.
\]

Jensen's inequality gives \(f_{\mathrm q}\geq f_{\mathrm a}\). Explain the mechanism immediately. Averaging \(Z_J\) allows realizations with unusually large partition functions to dominate.

The proposed introductory example is independent Ising spins in frozen Gaussian fields,

\[
H(s;h)=-\sum_{i=1}^N h_i s_i,
\qquad h_i\sim\mathcal N(0,\Delta^2).
\]

Everything is calculable,

\[
Z_h=\prod_i2\cosh(\beta h_i),
\]

\[
f_{\mathrm q}
=-\frac1\beta\mathbb E_h\log[2\cosh(\beta h)],
\qquad
f_{\mathrm a}
=-\frac{\log2}{\beta}-\frac{\beta\Delta^2}{2}.
\]

The annealed answer fails badly at low temperature even though the model has simple equilibrium structure. This separates the need to handle quenched averages from the additional question of RSB. These formulas are proposed derivations for the post. No experiment has been run for this draft.

### 3. Introduce replicas through that same example, including their mathematical footing

For positive integer \(n\), replicas are independent configurations sharing the same disorder. Averaging over the Gaussian fields gives

\[
\mathbb E_h Z_h^n
=\left[
\sum_{s^1,\ldots,s^n=\pm1}
\exp\left(
\frac{\beta^2\Delta^2}{2}\left[\sum_a s^a\right]^2
\right)
\right]^N.
\]

The shared disorder creates a coupling between replicas after averaging. That calculation gives the trick a concrete purpose.

Explain its mathematical status in three steps. First, \(Z_J\) must be positive and finite, and the desired disorder average must exist. Second, under suitable integrability,

\[
\left.\frac{d}{dn}\log\mathbb E_J Z_J^n\right|_{n=0}
=\mathbb E_J\log Z_J.
\]

Third, calculations performed only at integer \(n\) require a justified continuation toward zero. Integer values alone don't determine that continuation, and exchanging the \(n\to0\) and \(N\to\infty\) limits needs its own argument.

Jacob's Wishart log-determinant notes would be the next worked calculation because they connect this machinery directly to Gaussian integrals and Bayesian inference. Talagrand's proof of the Parisi formula supplies an example of a replica prediction subsequently established by rigorous methods. [Wishart notes](https://jzv.io/assets/pdf/replica_trick_wishart_log_det_notes.pdf), [Talagrand](https://annals.math.princeton.edu/2006/163-1/p04).

### 4. Freeze the dataset and let the parameters fluctuate

Start with

\[
p_\beta(\theta\mid D)
=\frac{p_0(\theta)e^{-\beta L_D(\theta)}}{Z_D(\beta)}.
\]

Replicas now mean independent parameter samples conditioned on the same dataset. Bayesian inference gives this ensemble a direct probabilistic interpretation when the prior and likelihood match the stated model.

Use Gaussian linear regression as the solvable anchor. Derive the posterior covariance, the log-determinant contribution to \(\log Z_D\), and predictive error. Random design matrices then bring Wishart theory into the story before a difficult neural-network calculation. From there, Li–Sompolinsky provides a route into deep linear networks through successive integration over layers. The official Les Houches page links Sompolinsky's lectures, while the paper provides a written derivation to work against. [Ridge-regression notes](https://jzv.io/assets/pdf/am226_generalization_in_ridge_regression_lecture_notes.pdf), [Li–Sompolinsky](https://arxiv.org/abs/2012.04030), [Les Houches course](https://leshouches2022.github.io/).

For RSB in this setting, add a controlled nonlinear example. The spherical perceptron's convex and nonconvex constraint regimes connect capacity, jamming, and fragmentation of the solution space. This gives RSB a specific explanatory job within the learning section. [Franz–Parisi](https://arxiv.org/abs/1501.03397).

### 5. Freeze the weights and explore the inputs

Define a family of input measures,

\[
p_{\beta,W}(x)
=\frac{p_0(x)e^{-\beta H_W(x)}}{Z_W(\beta)}.
\]

At \(\beta=0\), inputs follow the reference distribution \(p_0\). Increasing \(\beta\) emphasizes low-output regions of the chosen Hamiltonian. ARC's Gaussian-input expectation and Winer–Hanin's tilted input landscapes can then sit beside each other within one framework. [ARC](https://arxiv.org/abs/2605.05179v3), [Winer–Hanin](https://arxiv.org/html/2503.23982v2).

Derive the linear-network case first by completing the square. Then introduce input overlaps,

\[
Q_{ab}=\frac1d x^a\cdot x^b.
\]

Winer–Hanin's replica calculations provide the nonlinear comparison, including numerical evidence for full RSB with sinusoidal activations and RS behavior in the ReLU and tanh cases they investigate. The insight concerns how thermodynamically important inputs are organized.

### 6. Explain what changes when replica symmetry breaks

An RS ansatz assigns a common off-diagonal overlap \(Q_{ab}=q\). One-step and full RSB introduce progressively richer overlap structure. Connect that mathematical choice to how independently sampled configurations relate to one another within the same frozen realization.

Return briefly to the random-field example. Its replicas can have positive overlap because they respond to the same fields, while their overlap still concentrates as system size grows. This makes clear why shared preferences alone don't establish a glassy hierarchy.

For each neural-network example, pair the overlap calculation with a stability analysis of the RS solution. In experiments, examine overlap distributions within individual disorder realizations, account for ordinary symmetries, and check equilibration and size dependence before interpreting broad distributions as RSB evidence.

### 7. Finish by making the thermodynamic limits explicit

Give each calculation a short statement of what grows and what stays fixed. This prevents infinite width from absorbing several different theories.

| Calculation | Limit to specify |
|---|---|
| Random-initialization NNGP | Width grows at fixed depth and a fixed finite collection of probes |
| High-dimensional regression or perceptron | Sample count and input dimension grow together, with \(P/d\to\alpha\) |
| Winer–Hanin input landscapes | Input dimension and layer widths grow with specified ratios, using their Hamiltonian normalization |
| ARC estimation | Width grows at fixed depth, with the required estimation accuracy tracked explicitly |

The NNGP comparison belongs here because convergence on finitely many probes provides a different kind of information from a calculation over a growing input space. Jacob's notes make the fixed-probe limit explicit. Winer–Hanin also uses a readout normalization chosen for extensive thermodynamics, so reproducing its scaling should precede comparisons between experiments. [NNGP notes](https://jzv.io/assets/pdf/lecture_notes_on_nngp_from_mft.pdf), [Winer–Hanin definitions](https://arxiv.org/html/2503.23982v2#S3.SS1).

## Approved empirical figures

Brainstormed and approved November 21, 2026. The author selected 1–3 and prefers useful computed figures to the schematic point-mass/overlap-density plot. All three have been computed locally on CPU. No cloud compute was allocated. The original specifications follow; the implementation choices and numerical evidence are saved in [experiments/README.md](experiments/README.md) and the adjacent result directories.

The Bayesian figure uses eight paired width-512 hidden-feature draws and exact Gaussian readout conditioning. Adding hidden bias lowers mean grid RMSE from 0.707707 to 0.058396 for the deliberately incompatible target. The NNGP experiment uses fixed probes and independent random initializations; after seeing the static comparison, the author requested an animated scatter as width grows and removal of the MMD panel. Its animation couples actual network draws through prefix neuron sums up to width 1,024. The joint-discrepancy calculation remains saved as supplemental evidence, rather than a plotted panel. The replica figure uses a 20-step one-hidden-layer sine ansatz at explicit N/d=1, with integration and hierarchy refinements. Branching occurs between beta 3.5 and 3.75 on the chosen grid, consistent with the local RS instability. This adapts the published construction; it does not claim the same transition as Winer–Hanin's figure or a proof of full RSB. The proposed tanh control was not run; the exact linear model supplies the activation control.

### 1. Architectural symmetry as an inductive bias

Place in III after the Bayesian ensemble, with a return to its kernel interpretation in V. Use one-hidden-layer tanh random features on circle inputs x(t)=(cos t,sin t), comparing zero hidden biases with nonzero Gaussian hidden biases. Hold total preactivation variance and marginal output variance fixed across conditions. Fix 12 labels, preferably six antipodal pairs, from cos(2t)+0.3 sin(t). Freeze hidden features and do exact Gaussian Bayesian readout inference, explicitly labeling this restricted learning model rather than all-weight Bayesian training. A modest starting scope is width 512 and several paired feature realizations.

Plot prior variance by angular Fourier mode and posterior predictions (mean and uncertainty bands) from the same observations. Without biases, odd tanh forces f(t+pi)=-f(t), excluding even modes exactly; biases lift that restriction. This shows both a precise prior preference and its effect on inference. Do not present posterior extrapolation as a finite-network SGD result.

### 2. Finite-width networks approaching their NNGP

Place in V immediately after the central-limit explanation. Start with one hidden layer and ReLU, eight fixed non-antipodal probes, standard 1/sqrt(N) readout scaling, widths 8/32/128/512 and around 2,000 independent initializations per width. Hold probes, depth, input dimension, and distributions fixed. Reuse the circle input code, but avoid antipodal probe pairs if using a nonsingular covariance representation.

Plot paired-output clouds at small and large width against the theoretical GP covariance ellipse, with a companion joint-distribution discrepancy curve across width. Calibrate that discrepancy against equally sized GP-vs-GP samples to show its sampling floor. A covariance-error curve alone is insufficient: in the single-hidden-layer model the unconditional covariance already equals the limiting kernel at every width. A fourth-cumulant check is another option; for a scalar one-layer ReLU output with independent centered Gaussian readout weights and Gaussian preactivations, its standardized excess kurtosis is 15/N. At larger widths that statistic needs enough initialization samples to resolve it.

References: [Zavatone-Veth's notes](https://jzv.io/assets/pdf/lecture_notes_on_nngp_from_mft.pdf), [Matthews et al.](https://arxiv.org/abs/1804.11271).

### 3. A temperature-driven overlap bifurcation

Replace the overlap schematic in IV with a numerical-replica figure in the style of Winer–Hanin Fig. 2a. That figure plots overlap levels q_i(beta) from a 50-step RSB saddle-point calculation for a one-hidden-layer sine network at infinite width. It is not a direct Monte Carlo plot. Reproduce the same model and dimensional ratios, compare a tanh control, and check 10/20/50 RSB steps on a modest beta grid. Plot selected hierarchy levels as branches, with an optional companion curve q_max-q_min.

This is the most involved proposal because it needs a validated replica-equation solver. An exactly RS initialization stays RS under the zipper updates, so use symmetry-breaking perturbations and compare convergence, residuals, and relevant solution/stability criteria. Do not assume a universal critical beta or identify finite-size sampling quantiles with the limiting Parisi branches.

Reference: [Winer–Hanin Fig. 2 and zipper method](https://arxiv.org/html/2503.23982v2#S3.SS6).

### 4. Optional cheap figure for the ARC section

Place in VI before the discussion of ARC. Draw around 20 small one-hidden-layer ReLU networks and compute each exact Gaussian-input mean m_W=(1/sqrt(2pi)) sum_i a_i ||w_i||. Plot the instance means around the zero ensemble mean, then Monte Carlo estimation error versus input-sample count for those same frozen networks using the exact expression as ground truth. A useful small starting model is input dimension 8 and width 64. This is a solvable motivating example, not a reproduction of ARC's deep-network cumulant-propagation algorithm.

Implemented order: 1 and 2 first, followed by 3 as a dedicated numerical-theory calculation. Experiment 4 remains optional and unrun. Sources, seeds, normalization conventions, saved arrays, numerical checks, and exact source snapshots accompany the completed calculations.

## Earlier proposed experiments

Prioritize a small sequence in which each figure answers one question.

| Priority | Proposed experiment | Figure and purpose |
|---|---|---|
| 1 | Evaluate the random-field model using its exact formulas and direct samples | Quenched versus annealed free energy against temperature, alongside overlap concentration with increasing \(N\) |
| 2 | Solve Gaussian Bayesian regression while varying \(P/d\), noise, and regularization | Predictive error and posterior uncertainty against the corresponding random-matrix calculation |
| 3 | Sample inputs of fixed linear, ReLU, tanh, and sinusoidal networks under the specified Gibbs measures | Energy and within-network overlap distributions across temperature and system size |
| 4 | Reproduce one ARC expectation-estimation comparison using its released code | Estimation error against FLOPs, averaged over fixed network instances, connecting disorder theory to a computational benefit |
| Extension | Study the spherical perceptron across margin and constraint-density regimes | Feasibility and overlap geometry compared with the RS stability boundary |

The ARC experiment has [released code](https://github.com/alignment-research-center/mlp_cumulant_propagation). For the input-sampling experiment, multiple equilibrated chains and a linear exact check are essential because sampling failure can otherwise resemble fragmentation.

The proposed core is the random-field derivation, Gaussian regression, and fixed-network input landscape, with ARC as the practical payoff. A full perceptron RSB derivation can be an extension if the post needs a deeper treatment of glassy phases.

## Reference catalog

These entries preserve the references cited above and the additional sources used while drafting the opening. Links identify the source and its intended use. Access notes distinguish material inspected from resources retained for later reading.

### Neural networks and replicas

- Mike Winer and Boris Hanin, *Deep Neural Nets as Hamiltonians*, arXiv:2503.23982. [Abstract and versions](https://arxiv.org/abs/2503.23982), [v2 full text used in planning](https://arxiv.org/html/2503.23982v2), [published article](https://doi.org/10.1103/xg33-ksqn). Frozen-weight input landscapes, extensive scaling, overlap equations, and numerical RSB analysis. Distinguish the replica calculation and its numerical solutions from a rigorous proof for every architecture.
- Wilson Wu, Victor Lecomte, Michael Winer, George Robinson, Jacob Hilton, and Paul Christiano, *Estimating the expected output of wide random MLPs more efficiently than sampling*, arXiv:2605.05179v3, revised November 2, 2026. [Versioned paper](https://arxiv.org/abs/2605.05179v3), [full text](https://arxiv.org/html/2605.05179v3), [code](https://github.com/alignment-research-center/mlp_cumulant_propagation). Fixed-instance expectations over inputs and average-case error guarantees over weights.
- Jacob Hilton, *Mechanistic estimation for wide random MLPs*, May 7, 2026. [ARC explanation](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/). Accessible account of the estimation problem and its computational motivation.
- Michael Winer, *A Mike's-Eye View of ARC's Research*, June 9, 2026. [ARC post](https://www.alignment.org/blog/a-mikes-eye-view-of-arcs-research/). Broader motivation retained for later reading and possible context.
- Jacob A. Zavatone-Veth, [notes index](https://jzv.io/notes). Entry point for the following expository resources.
- Zavatone-Veth, *What is the simplest model problem for the replica trick in random matrix theory?*, November 10, 2024. [Wishart log-determinant PDF](https://jzv.io/assets/pdf/replica_trick_wishart_log_det_notes.pdf). Gaussian partition function, replica calculation, and a solvable link to Bayesian inference.
- Zavatone-Veth, *Lecture notes on the replica method for Wishart matrix eigenvalues*. [PDF](https://jzv.io/assets/pdf/wishart_eigenvalue_and_gaussian_data_pca_replica_notes.pdf). Original notes from September 2022, with the inspected PDF dated November 27, 2025. Replica-symmetric calculation of spectral edges and proportional asymptotics.
- Zavatone-Veth, *Lecture notes on the inductive biases of high-dimensional ridge regression*. [PDF](https://jzv.io/assets/pdf/am226_generalization_in_ridge_regression_lecture_notes.pdf). Generalization and random-design calculations.
- Zavatone-Veth, *Lecture notes on Gaussian process behavior in wide neural networks*. [PDF](https://jzv.io/assets/pdf/lecture_notes_on_nngp_from_mft.pdf). Fixed-probe NNGP limit and mean-field derivation.
- Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein, *Deep Neural Networks as Gaussian Processes*, ICLR 2018. [Paper](https://arxiv.org/abs/1711.00165). Gaussian-process correspondence and Bayesian prediction, used in section V.
- Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, and Zoubin Ghahramani, *Gaussian Process Behaviour in Wide Deep Neural Networks*, 2018. [Paper](https://arxiv.org/abs/1804.11271). Convergence of wide deep networks under stated regularity conditions, used in section V and appendix C.
- Zavatone-Veth, *Expository notes on the storage capacity of perceptrons with constrained weight distributions*. [PDF](https://jzv.io/assets/pdf/perceptron_constrained_weight_distribution_notes.pdf). Retained for the storage-capacity extension.
- Qianyi Li and Haim Sompolinsky, *Statistical Mechanics of Deep Linear Neural Networks: The Backpropagating Kernel Renormalization*, Physical Review X 11, 031059 (2021). [Preprint](https://arxiv.org/abs/2012.04030), [v2 full text used in section III](https://arxiv.org/html/2012.04030v2), [published article](https://doi.org/10.1103/PhysRevX.11.031059). Integration over learned weights for a fixed dataset.
- *Summer school on Statistical Physics & Machine Learning*, Les Houches, July 4–29, 2022. [Official course page](https://leshouches2022.github.io/), [Sompolinsky video playlist](https://www.youtube.com/watch?ab_channel=LesHouches2022&list=PLEIq5bchE3R2MM_qCYzDS_b4uOZQXdrBj&v=sJswk8JhdOk). The course listing was inspected. The videos still need to be worked through. The files titled “Les Houches 0722 – Lecture 1/2” linked elsewhere on the page belong to Sara Solla's course and should retain that attribution.
- Silvio Franz and Giorgio Parisi, *The simplest model of jamming*, arXiv:1501.03397. [Preprint](https://arxiv.org/abs/1501.03397), [v1 full text used in section III](https://arxiv.org/html/1501.03397v1). Convex and nonconvex perceptron regimes, jamming, and RSB.
- Lenka Zdeborová and Florent Krzakala, *Statistical physics of inference: Thresholds and algorithms*, Advances in Physics 65, 453–552 (2016), revised 2018. [Preprint](https://arxiv.org/abs/1511.02476), [v5 full text used in section III](https://arxiv.org/html/1511.02476v5). Matched Bayesian inference, exchangeability of generating parameters and posterior replicas, and Nishimori identities. The footnote states the exchangeability property without making a universal claim about the absence of RSB.
- Michel Talagrand, *The Parisi formula*, Annals of Mathematics 163 (2006), 221–263. [Journal page](https://annals.math.princeton.edu/2006/163-1/p04), [DOI](https://doi.org/10.4007/annals.2006.163.221). Rigorous establishment of the spin-glass free-energy formula.

### Metallurgy, spin glasses, and the opening narrative

- H. K. D. H. Bhadeshia, [Cambridge hardenability practical](https://www.phase-trans.msm.cam.ac.uk/2000/practicals/AP3/AP3.html). Cooling rates, diffusion, martensite formation, and tempering.
- Bhadeshia, *Ferrous Alloys*, Course MP10, Lecture 6. [PDF](https://www.phase-trans.msm.cam.ac.uk/2002/MP10-6.pdf), especially PDF pages 1–3. Austenite, trapped carbon, tetragonal strain, dislocation motion, and tempering.
- Bhadeshia, *Tempered Martensite*. [Article](https://www.phase-trans.msm.cam.ac.uk/2004/Tempered.Martensite/tempered.martensite.html). Metastability, carbon mobility, carbide precipitation, and recovery of defects.
- Bhadeshia, *Martensite and Martensitic Phase Transformations*. [Resource collection](https://www.phase-trans.msm.cam.ac.uk/2002/martensite.html). Coordinated lattice transformation and crystallography. Introductory lectures are linked from the collection.
- Samuel J. Ling, Jeff Sanny, and William Moebs, *University Physics, Volume 2*, section 12.7, *Magnetism in Matter*. [OpenStax chapter](https://openstax.org/books/university-physics-volume-2/pages/12-7-magnetism-in-matter). Accessible definition of an atomic magnetic moment for the opening.
- Marc Mézard and Andrea Montanari, *Information, Physics, and Computation*, Part B, section 5.4. [Author-hosted draft PDF](https://web.stanford.edu/~montanar/RESEARCH/BOOK/partB.pdf), printed pages 102–103, PDF pages 10–11. Explicit explanation of the metallurgical origin of “quenched” and “annealed,” separate thermal/disorder averages, and rare-sample domination. The hosted file is a November 2007 draft of the later book.
- S. F. Edwards and P. W. Anderson, *Theory of spin glasses*, Journal of Physics F 5 (1975), 965–974. [DOI](https://doi.org/10.1088/0305-4608/5/5/017). Original dilute-alloy model. Publisher full-text access was blocked during drafting. Use the author retrospective and accessible lectures for directly checked explanatory claims.
- Sam Edwards, retrospective on *Theory of spin glasses*, written July 17, 1985, published in *This Week's Citation Classic*, September 16, 1985. [One-page PDF](https://garfield.library.upenn.edu/classics1985/A1985APZ5700001.pdf). First-person account of magnetic interactions and the transition from rubber replicas to spin-glass replicas.
- Th. M. Nieuwenhuizen and C. N. A. van Duin, *Theory of site-disordered magnets*, arXiv:cond-mat/9804143 (1998). [Preprint](https://arxiv.org/abs/cond-mat/9804143). Random atomic positions in CuMn and related alloys determine magnetic couplings. Abstract inspected.
- Chandan Dasgupta, *Introduction to the Theory of Spin Glasses*, School on Glass Formers and Glasses, JNCASR, January 2010. [Lecture PDF](https://www.icts.res.in/sites/default/files/1267685100Chandan-Dasgupta-Jan8-Lecture1.pdf). CuMn, competing interactions, spin freezing, and the progression to replicas and RSB.
- David Sherrington and Scott Kirkpatrick, *Solvable Model of a Spin-Glass*, Physical Review Letters 35 (1975), 1792. [DOI](https://doi.org/10.1103/PhysRevLett.35.1792). Mean-field model with random couplings. Publisher abstract inspected.
- K. Binder and A. P. Young, *Spin glasses: Experimental facts, theoretical concepts, and open questions*, Reviews of Modern Physics 58 (1986), 801. [DOI](https://doi.org/10.1103/RevModPhys.58.801). Background reference for experiments, frustration, dynamics, and RSB. Publisher abstract inspected, full review retained for later reading.
- P. V. E. McClintock, *Ordered disorder?*, Nature 256 (1975), 369. [DOI](https://doi.org/10.1038/256369a0). Contemporary explanatory context retained from planning. The accessible publisher page provides metadata, while full-text access is restricted.

### Hero image

- Diego Velázquez, *The Forge of Vulcan* (1630), Museo Nacional del Prado. [Prado collection record](https://www.museodelprado.es/en/the-collection/art-work/vulcans-forge/84a0240d-b41a-404d-8433-6e4e2efd21ab). The collection page returned HTTP 403 during planning.
- *Juan de Pareja*, Metropolitan Museum of Art Bulletin 29, no. 10, part 2 (June 1971). [Museum PDF](https://resources.metmuseum.org/resources/metpublications/pdf/Juan_de_Pareja_by_Diego_Velazquez_The_Metropolitan_Museum_of_Art_Bulletin_v_29_no_10_Part_II_June_1971.pdf). Museum attribution and reproduction of the proposed painting.
- [Wikimedia Commons image record](https://commons.wikimedia.org/wiki/File%3AVel%C3%A1zquez_-_La_Fragua_de_Vulcano_%28Museo_del_Prado%2C_1630%29.jpg). Candidate reproduction source retained for later selection and attribution checking.
