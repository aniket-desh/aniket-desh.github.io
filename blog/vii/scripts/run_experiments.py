#!/usr/bin/env python3
"""Run the synthetic experiments for Post VII.

The script keeps the numerical model deliberately small: Gaussian linear
regression, eigendecompositions reused across every ridge value, and plain
NumPy/SciPy arrays.  It writes three compressed result files consumed by the
plotting script in the same directory.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

A = 1.0
ETA2 = 0.5


def unit_teacher(rng: np.random.Generator, d: int) -> NDArray[np.float64]:
    theta = rng.standard_normal(d)
    return theta / np.linalg.norm(theta)


def stable_mp_g(lam: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """Marchenko-Pastur resolvent for d / p -> q, evaluated stably."""
    lam = np.asarray(lam, dtype=float)
    a = lam + 1.0 - q
    root = np.sqrt(a * a + 4.0 * q * lam)
    return np.where(a >= 0.0, 2.0 / (root + a), (root - a) / (2.0 * q * lam))


def stable_mp_g_prime(lam: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    g = stable_mp_g(lam, q)
    denominator = -1.0 / (g * g) + q / (1.0 + q * g) ** 2
    return 1.0 / denominator


def mp_overlap(lambdas: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """Deterministic high-dimensional excess-error overlap matrix."""
    lambdas = np.asarray(lambdas, dtype=float)
    g = stable_mp_g(lambdas, q)
    gp = stable_mp_g_prime(lambdas, q)
    m = len(lambdas)
    result = np.empty((m, m), dtype=float)

    for i, lam_i in enumerate(lambdas):
        for j, lam_j in enumerate(lambdas):
            if i == j or np.isclose(lam_i, lam_j, rtol=0.0, atol=1e-14):
                h = -gp[i]
                k = g[i] + lam_i * gp[i]
            else:
                h = (g[i] - g[j]) / (lam_j - lam_i)
                k = (lam_j * g[j] - lam_i * g[i]) / (lam_j - lam_i)
            result[i, j] = A * lam_i * lam_j * h + ETA2 * q * k

    result = 0.5 * (result + result.T)
    if np.linalg.eigvalsh(result).min() < -1e-9:
        raise RuntimeError("MP overlap matrix is not positive semidefinite")
    return result


def conditional_overlap(
    spectrum: NDArray[np.float64],
    teacher_eig: NDArray[np.float64],
    lambdas: NDArray[np.float64],
    p: int,
) -> NDArray[np.float64]:
    denominators = spectrum[:, None] + lambdas[None, :]
    signal = lambdas[None, :] * teacher_eig[:, None] / denominators
    noise = np.sqrt(ETA2 / p) * np.sqrt(np.maximum(spectrum, 0.0))[:, None] / denominators
    result = signal.T @ signal + noise.T @ noise
    return 0.5 * (result + result.T)


def simplex_qp(
    hessian: NDArray[np.float64],
    linear: NDArray[np.float64] | None = None,
    initial: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Minimize w^T H w - 2 c^T w over the probability simplex."""
    hessian = 0.5 * (hessian + hessian.T)
    m = hessian.shape[0]
    linear = np.zeros(m) if linear is None else np.asarray(linear, dtype=float)
    initial = np.full(m, 1.0 / m) if initial is None else np.asarray(initial, dtype=float)

    def objective(weights: NDArray[np.float64]) -> float:
        return float(weights @ hessian @ weights - 2.0 * linear @ weights)

    def gradient(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2.0 * (hessian @ weights - linear)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0), "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 800, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"simplex optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, None)
    weights /= weights.sum()
    if abs(weights.sum() - 1.0) > 1e-10 or weights.min() < -1e-12:
        raise RuntimeError("invalid simplex weights")
    return weights


def topk(weights: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    if k >= len(weights):
        return weights / weights.sum()
    chosen = np.argpartition(weights, -k)[-k:]
    truncated = np.zeros_like(weights)
    truncated[chosen] = weights[chosen]
    if truncated.sum() <= 1e-15:
        truncated[chosen] = 1.0
    return truncated / truncated.sum()


def fit_simplex(predictions: NDArray[np.float64], targets: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(targets)
    hessian = predictions.T @ predictions / n
    linear = predictions.T @ targets / n
    return simplex_qp(hessian, linear)


def normalized_overlap(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    scale = np.sqrt(np.maximum(np.diag(matrix), 1e-30))
    return matrix / np.outer(scale, scale)


def validate_theory() -> None:
    for q in (0.5, 0.9, 1.1, 2.0):
        lambdas = np.geomspace(1e-4, 10.0, 40)
        g = stable_mp_g(lambdas, q)
        gp = stable_mp_g_prime(lambdas, q)
        residual = q * lambdas * g * g + (lambdas + 1.0 - q) * g - 1.0
        if np.max(abs(residual)) > 2e-10 or np.any(g <= 0.0) or np.any(gp >= 0.0):
            raise RuntimeError("Marchenko-Pastur resolvent validation failed")

        lam = 0.37 * q
        close = np.array([lam, lam * (1.0 + 1e-6)])
        overlap = mp_overlap(close, q)
        if abs(overlap[0, 1] - overlap[0, 0]) / overlap[0, 0] > 2e-5:
            raise RuntimeError("off-diagonal overlap does not approach its diagonal limit")


def run_overlap_experiment(*, quick: bool) -> None:
    q_values = np.array([0.5, 0.9, 1.1, 2.0])
    d_values = np.array([128, 256] if quick else [128, 256, 512])
    ratios = np.array([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16], dtype=float)
    n_design = 8 if quick else 30
    n_noise = 2 if quick else 8
    m = len(ratios)

    empirical = np.zeros((len(q_values), len(d_values), m, m))
    conditional = np.zeros_like(empirical)
    theory = np.zeros_like(empirical)
    q_effective = np.zeros((len(q_values), len(d_values)))
    relative_samples = np.zeros((len(q_values), len(d_values), n_design))

    for qi, q in enumerate(q_values):
        for di, d in enumerate(d_values):
            p = int(round(d / q))
            q_eff = d / p
            q_effective[qi, di] = q_eff
            lambdas = ratios * (q_eff * ETA2 / A)
            theory_here = mp_overlap(lambdas, q_eff)
            theory[qi, di] = theory_here
            realized_sum = np.zeros((m, m))
            conditional_sum = np.zeros((m, m))
            for seed in range(n_design):
                rng = np.random.default_rng(10_000 * qi + 1_000 * di + seed)
                theta = unit_teacher(rng, d)
                x = rng.standard_normal((p, d))
                covariance = x.T @ x / p
                spectrum, basis = np.linalg.eigh(covariance)
                teacher_eig = basis.T @ theta
                cond = conditional_overlap(spectrum, teacher_eig, lambdas, p)
                conditional_sum += cond
                relative_samples[qi, di, seed] = np.linalg.norm(cond - theory_here) / np.linalg.norm(
                    theory_here
                )

                noiseless_b = covariance @ theta
                for _ in range(n_noise):
                    noise = math.sqrt(ETA2) * rng.standard_normal(p)
                    b = noiseless_b + x.T @ noise / p
                    b_eig = basis.T @ b
                    estimates = b_eig[:, None] / (spectrum[:, None] + lambdas[None, :])
                    errors = estimates - teacher_eig[:, None]
                    realized_sum += errors.T @ errors

            empirical[qi, di] = realized_sum / (n_design * n_noise)
            conditional[qi, di] = conditional_sum / n_design

    np.savez_compressed(
        DATA / "overlap_results.npz",
        q_values=q_values,
        q_effective=q_effective,
        d_values=d_values,
        ratios=ratios,
        theory=theory,
        empirical=empirical,
        conditional=conditional,
        relative_mean=relative_samples.mean(axis=2),
        relative_se=relative_samples.std(axis=2, ddof=1) / math.sqrt(n_design),
        n_design=n_design,
        n_noise=n_noise,
    )


def run_weighting_experiment(*, quick: bool) -> None:
    d = 256
    q = 0.8
    p = int(round(d / q))
    lambda_star = q * ETA2 / A
    coarse_ratios = np.array([1 / 16, 1 / 8, 1 / 4, 1 / 2, 2, 4, 8, 16], dtype=float)
    coarse = coarse_ratios * lambda_star
    augmented = np.sort(np.append(coarse, lambda_star))
    k_values = np.array([1, 2, 4, 8])

    r_coarse = mp_overlap(coarse, q)
    r_augmented = mp_overlap(augmented, q)
    w_coarse = simplex_qp(r_coarse)
    w_augmented = simplex_qp(r_augmented)
    oracle_index = int(np.argmin(abs(augmented - lambda_star)))
    if w_augmented[oracle_index] < 0.999:
        raise RuntimeError("adding oracle ridge did not collapse the population weights")

    oracle_excess = float(mp_overlap(np.array([lambda_star]), q)[0, 0])
    fixed_coarse = np.array([topk(w_coarse, int(k)) @ r_coarse @ topk(w_coarse, int(k)) for k in k_values])
    fixed_augmented = np.array(
        [topk(w_augmented, int(k)) @ r_augmented @ topk(w_augmented, int(k)) for k in k_values]
    )
    quality_order = np.argsort(np.diag(r_coarse))
    quality_coarse = []
    for k in k_values:
        weights = np.zeros(len(coarse))
        weights[quality_order[: int(k)]] = 1.0 / k
        quality_coarse.append(weights @ r_coarse @ weights)
    quality_coarse = np.asarray(quality_coarse)
    if fixed_coarse.min() < oracle_excess - 1e-8:
        raise RuntimeError("fixed population weights beat oracle ridge")

    fit_sizes = np.array([32, 128, 512, 2048])
    n_seed = 16 if quick else 100
    n_fit = len(fit_sizes)
    n_k = len(k_values)
    m = len(coarse)
    adaptive_risk = np.zeros((n_seed, n_fit, n_k))
    greedy_risk = np.zeros_like(adaptive_risk)
    fixed_risk = np.zeros_like(adaptive_risk)
    tuned_ridge_risk = np.zeros((n_seed, n_fit))
    train_plus_fit_risk = np.zeros((n_seed, n_fit))
    adaptive_weights = np.zeros((n_seed, n_fit, m))
    individual_risk = np.zeros((n_seed, m))
    train_oracle_risk = np.zeros(n_seed)
    dense_lambdas = lambda_star * np.geomspace(1 / 32, 32, 129)

    fixed_by_k = np.stack([topk(w_coarse, int(k)) for k in k_values])
    for seed in range(n_seed):
        rng = np.random.default_rng(200_000 + seed)
        theta = unit_teacher(rng, d)
        x = rng.standard_normal((p, d))
        epsilon = math.sqrt(ETA2) * rng.standard_normal(p)
        y = x @ theta + epsilon
        covariance = x.T @ x / p
        b = x.T @ y / p
        spectrum, basis = np.linalg.eigh(covariance)
        b_eig = basis.T @ b
        teacher_eig = basis.T @ theta
        estimates_eig = b_eig[:, None] / (spectrum[:, None] + coarse[None, :])
        errors_eig = estimates_eig - teacher_eig[:, None]
        individual_risk[seed] = ETA2 + np.sum(errors_eig * errors_eig, axis=0)
        oracle_estimate = b_eig / (spectrum + lambda_star)
        train_oracle_risk[seed] = ETA2 + np.sum((oracle_estimate - teacher_eig) ** 2)

        estimates = basis @ estimates_eig
        dense_estimates = basis @ (b_eig[:, None] / (spectrum[:, None] + dense_lambdas[None, :]))
        x_fit = rng.standard_normal((int(fit_sizes[-1]), d))
        y_fit = x_fit @ theta + math.sqrt(ETA2) * rng.standard_normal(int(fit_sizes[-1]))
        pool_predictions = x_fit @ estimates
        dense_predictions = x_fit @ dense_estimates

        for fi, size in enumerate(fit_sizes):
            size = int(size)
            pred = pool_predictions[:size]
            targets = y_fit[:size]
            weights = fit_simplex(pred, targets)
            adaptive_weights[seed, fi] = weights
            fitness_losses = np.mean((pred - targets[:, None]) ** 2, axis=0)
            greedy_order = np.argsort(fitness_losses)

            for ki, k in enumerate(k_values):
                learned = topk(weights, int(k))
                error = errors_eig @ learned
                adaptive_risk[seed, fi, ki] = ETA2 + error @ error

                greedy = np.zeros(m)
                greedy[greedy_order[: int(k)]] = 1.0 / k
                error = errors_eig @ greedy
                greedy_risk[seed, fi, ki] = ETA2 + error @ error

                error = errors_eig @ fixed_by_k[ki]
                fixed_risk[seed, fi, ki] = ETA2 + error @ error

            dense_loss = np.mean((dense_predictions[:size] - targets[:, None]) ** 2, axis=0)
            chosen = int(np.argmin(dense_loss))
            dense_error = dense_estimates[:, chosen] - theta
            tuned_ridge_risk[seed, fi] = ETA2 + dense_error @ dense_error

            combined_p = p + size
            covariance_combined = (x.T @ x + x_fit[:size].T @ x_fit[:size]) / combined_p
            b_combined = (x.T @ y + x_fit[:size].T @ targets) / combined_p
            q_combined = d / combined_p
            combined_lambda = q_combined * ETA2 / A
            combined_estimate = np.linalg.solve(
                covariance_combined + combined_lambda * np.eye(d), b_combined
            )
            combined_error = combined_estimate - theta
            train_plus_fit_risk[seed, fi] = ETA2 + combined_error @ combined_error

    np.savez_compressed(
        DATA / "weighting_results.npz",
        d=d,
        q=q,
        p=p,
        lambda_star=lambda_star,
        coarse_ratios=coarse_ratios,
        coarse=coarse,
        augmented=augmented,
        k_values=k_values,
        fit_sizes=fit_sizes,
        r_coarse=r_coarse,
        w_coarse=w_coarse,
        w_augmented=w_augmented,
        fixed_coarse=ETA2 + fixed_coarse,
        fixed_augmented=ETA2 + fixed_augmented,
        quality_coarse=ETA2 + quality_coarse,
        oracle_risk=ETA2 + oracle_excess,
        adaptive_risk=adaptive_risk,
        greedy_risk=greedy_risk,
        fixed_risk=fixed_risk,
        tuned_ridge_risk=tuned_ridge_risk,
        train_plus_fit_risk=train_plus_fit_risk,
        adaptive_weights=adaptive_weights,
        individual_risk=individual_risk,
        train_oracle_risk=train_oracle_risk,
        n_seed=n_seed,
    )


def pair_correlations(errors: NDArray[np.float64], trajectories: int, cycles: int) -> tuple[float, float]:
    norms = np.linalg.norm(errors, axis=1)
    correlation = errors @ errors.T / np.outer(norms, norms)
    within: list[float] = []
    cross: list[float] = []
    for i in range(len(errors)):
        trajectory_i = i // cycles
        for j in range(i + 1, len(errors)):
            if trajectory_i == j // cycles:
                within.append(float(correlation[i, j]))
            else:
                cross.append(float(correlation[i, j]))
    return float(np.mean(within)), float(np.mean(cross))


def run_chain_experiment(*, quick: bool) -> None:
    d = 256
    q = 0.8
    p = int(round(d / q))
    lambda_star = q * ETA2 / A
    n_seed = 8 if quick else 50
    exact_cycles = 32

    exact_distance = np.zeros((n_seed, exact_cycles + 1))
    exact_risk = np.zeros_like(exact_distance)
    exact_alignment = np.zeros_like(exact_distance)

    alpha_exact = 0.5
    lambda_exact = (1.0 - alpha_exact) * lambda_star
    for seed in range(n_seed):
        rng = np.random.default_rng(300_000 + seed)
        theta = unit_teacher(rng, d)
        x = rng.standard_normal((p, d))
        y = x @ theta + math.sqrt(ETA2) * rng.standard_normal(p)
        covariance = x.T @ x / p
        b = x.T @ y / p
        spectrum, basis = np.linalg.eigh(covariance)
        b_eig = basis.T @ b
        teacher_eig = basis.T @ theta
        limit = b_eig / (spectrum + lambda_star)
        state = np.zeros(d)
        initial_distance = np.sum((state - limit) ** 2)
        limit_error = limit - teacher_eig
        for cycle in range(exact_cycles + 1):
            error = state - teacher_eig
            exact_distance[seed, cycle] = np.sum((state - limit) ** 2) / initial_distance
            exact_risk[seed, cycle] = ETA2 + error @ error
            exact_alignment[seed, cycle] = (error @ limit_error) / (
                np.linalg.norm(error) * np.linalg.norm(limit_error)
            )
            if cycle < exact_cycles:
                target = (1.0 - alpha_exact) * b_eig + alpha_exact * spectrum * state
                next_state = target / (spectrum + lambda_exact)
                contraction = alpha_exact * spectrum / (spectrum + lambda_exact)
                closed = limit + contraction ** (cycle + 1) * (np.zeros(d) - limit)
                if np.max(abs(next_state - closed)) > 1e-10:
                    raise RuntimeError("exact chain does not match its closed form")
                state = next_state

    alphas = np.array([0.0, 0.5])
    # q0 begins near 0.25 and ends near 0.05, so sigma is the initial
    # relative kick and the schedule below decays it to one fifth.
    sigmas = np.array([0.0, 0.10, 0.25])
    steps = np.array([1, 4, 16])
    configs = np.array([(a, s, int(step)) for a in alphas for s in sigmas for step in steps])
    trajectories = 4
    cycles = 16
    tail_cycles = 4
    candidates = trajectories * tail_cycles
    top_k = 8
    fit_size = 512
    metrics = {
        name: np.zeros((n_seed, len(configs)))
        for name in (
            "individual_risk",
            "within_correlation",
            "cross_correlation",
            "effective_rank",
            "learned_risk",
            "greedy_risk",
            "best_risk",
        )
    }

    for seed in range(n_seed):
        rng = np.random.default_rng(400_000 + seed)
        theta = unit_teacher(rng, d)
        x = rng.standard_normal((p, d))
        y = x @ theta + math.sqrt(ETA2) * rng.standard_normal(p)
        covariance = x.T @ x / p
        b = x.T @ y / p
        spectrum, basis = np.linalg.eigh(covariance)
        b_eig = basis.T @ b
        teacher_eig = basis.T @ theta
        x_fit = rng.standard_normal((fit_size, d))
        y_fit = x_fit @ theta + math.sqrt(ETA2) * rng.standard_normal(fit_size)
        kicks = rng.standard_normal((trajectories, cycles, d))

        for ci, (alpha, sigma, step_count_float) in enumerate(configs):
            step_count = int(step_count_float)
            lam = (1.0 - alpha) * lambda_star
            gamma = 0.9 / (spectrum.max() + lam)
            history = np.zeros((trajectories, cycles, d))
            for trajectory in range(trajectories):
                previous = np.zeros(d)
                for cycle in range(cycles):
                    if cycles == 1:
                        decay = 1.0
                    else:
                        decay = 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * cycle / (cycles - 1)))
                    scale = sigma * decay * np.linalg.norm(previous) / math.sqrt(d)
                    start = previous + scale * kicks[trajectory, cycle]
                    target = (1.0 - alpha) * b_eig + alpha * spectrum * previous
                    optimum = target / (spectrum + lam)
                    contraction = (1.0 - gamma * (spectrum + lam)) ** step_count
                    state = optimum + contraction * (start - optimum)
                    history[trajectory, cycle] = state
                    previous = state

            tail = history[:, -tail_cycles:, :].reshape(candidates, d)
            errors = tail - teacher_eig[None, :]
            gram = errors @ errors.T
            within, cross = pair_correlations(errors, trajectories, tail_cycles)
            eigenvalues = np.linalg.eigvalsh(gram)
            effective_rank = eigenvalues.sum() ** 2 / np.sum(eigenvalues * eigenvalues)

            estimates = basis @ tail.T
            predictions = x_fit @ estimates
            learned = fit_simplex(predictions, y_fit)
            learned = topk(learned, top_k)
            learned_error = errors.T @ learned
            fitness_loss = np.mean((predictions - y_fit[:, None]) ** 2, axis=0)
            greedy_order = np.argsort(fitness_loss)
            greedy = np.zeros(candidates)
            greedy[greedy_order[:top_k]] = 1.0 / top_k
            greedy_error = errors.T @ greedy

            metrics["individual_risk"][seed, ci] = ETA2 + np.mean(np.diag(gram))
            metrics["within_correlation"][seed, ci] = within
            metrics["cross_correlation"][seed, ci] = cross
            metrics["effective_rank"][seed, ci] = effective_rank
            metrics["learned_risk"][seed, ci] = ETA2 + learned_error @ learned_error
            metrics["greedy_risk"][seed, ci] = ETA2 + greedy_error @ greedy_error
            metrics["best_risk"][seed, ci] = ETA2 + np.diag(gram).min()

    np.savez_compressed(
        DATA / "chain_results.npz",
        d=d,
        q=q,
        p=p,
        lambda_star=lambda_star,
        exact_cycles=np.arange(exact_cycles + 1),
        exact_distance=exact_distance,
        exact_risk=exact_risk,
        exact_alignment=exact_alignment,
        configs=configs,
        alphas=alphas,
        sigmas=sigmas,
        steps=steps,
        trajectories=trajectories,
        cycles=cycles,
        tail_cycles=tail_cycles,
        top_k=top_k,
        fit_size=fit_size,
        n_seed=n_seed,
        **metrics,
    )


def write_manifest(*, quick: bool, elapsed: float) -> None:
    manifest = {
        "quick": quick,
        "elapsed_seconds": elapsed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed_policy": "fixed independent NumPy generators by experiment and repetition",
        "model": {"teacher_norm_squared": A, "label_noise_variance": ETA2},
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    parser.add_argument(
        "--experiment",
        choices=("all", "overlap", "weighting", "chain"),
        default="all",
    )
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validate_theory()

    if args.experiment in ("all", "overlap"):
        print("running overlap experiment", flush=True)
        run_overlap_experiment(quick=args.quick)
    if args.experiment in ("all", "weighting"):
        print("running weighting experiment", flush=True)
        run_weighting_experiment(quick=args.quick)
    if args.experiment in ("all", "chain"):
        print("running chain experiment", flush=True)
        run_chain_experiment(quick=args.quick)

    elapsed = time.perf_counter() - started
    write_manifest(quick=args.quick, elapsed=elapsed)
    print(f"finished in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
