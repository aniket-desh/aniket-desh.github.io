"""An exact Bayesian readout experiment exposing an architectural symmetry.

Hidden tanh features are frozen.  Only the Gaussian readout is inferred.
Run this file to compute and plot; --plot-only regenerates from saved arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
from scipy.special import roots_hermitenorm

from plot_style import ROOT, configure, save_figure
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results' / 'inductive-bias'
CONFIG = {
    'width': 512,
    'grid_points': 512,
    'observations': 12,
    'data_seed': 20261121,
    'feature_seeds': list(range(100, 108)),
    'representative_seed': 100,
    'noise_sd': 0.05,
    'bias_variances': [0.0, 0.25],
    'weight_variances': [1.0, 0.75],
    'readout_prior': 'independent standard Gaussian',
    'feature_normalization': '1 / sqrt(width * E[tanh(Z)^2]), Z ~ N(0,1)',
    'quadrature_order': 192,
    'target': 'cos(2*t) + 0.3*sin(t)',
    'posterior_band': 'latent mean +/- 2 posterior standard deviations',
    'learning_scope': 'exact Gaussian posterior over readout; hidden weights and biases frozen',
}


def target(t):
    return np.cos(2 * t) + 0.3 * np.sin(t)


def circle(t):
    return np.column_stack((np.cos(t), np.sin(t)))


def tanh_second_moment(order):
    nodes, weights = roots_hermitenorm(order)
    return float(np.dot(weights, np.tanh(nodes) ** 2) / np.sqrt(2 * np.pi))


def feature_matrix(x, base_w, base_b, bias_variance, normalizer):
    # Unit-circle inputs make every preactivation's ensemble variance exactly one.
    weight_variance = 1.0 - bias_variance
    preactivation = np.sqrt(weight_variance) * (x @ base_w.T)
    preactivation += np.sqrt(bias_variance) * base_b[None, :]
    return np.tanh(preactivation) / np.sqrt(base_w.shape[0] * normalizer)


def posterior(phi_train, phi_test, y, noise_sd):
    # The n_train x n_train solve is equivalent to the full readout-weight posterior.
    k_train = phi_train @ phi_train.T
    k_cross = phi_test @ phi_train.T
    factor = cho_factor(k_train + noise_sd ** 2 * np.eye(len(y)), lower=True)
    mean = k_cross @ cho_solve(factor, y)
    variance = np.sum(phi_test ** 2, axis=1)
    variance -= np.einsum('ij,ji->i', k_cross, cho_solve(factor, k_cross.T))
    assert variance.min() > -1e-10, variance.min()
    return mean, np.maximum(variance, 0), k_train


def mode_variances(phi):
    # For independent N(0,1) readout weights, sum_i |FFT(phi_i)|^2 is
    # E_readout |FFT(f)|^2.  Double interior modes for a one-sided real spectrum.
    coefficients = np.fft.rfft(phi, axis=0) / len(phi)
    power = np.sum(np.abs(coefficients) ** 2, axis=1)
    power[1:-1] *= 2
    return power


def independent_checks(normalizer):
    reference = quad(
        lambda z: np.tanh(z) ** 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
        -12, 12, epsabs=1e-13, epsrel=1e-13,
    )[0]
    normalizer_error = abs(normalizer - reference)
    assert normalizer_error < 1e-9, normalizer_error

    # Independent small dense weight-space reference for the kernel conditioning.
    rng = np.random.default_rng(24)
    train = rng.normal(size=(5, 9)) / 3
    test = rng.normal(size=(7, 9)) / 3
    y = rng.normal(size=5)
    sigma = 0.2
    m_kernel, v_kernel, _ = posterior(train, test, y, sigma)
    covariance = np.linalg.inv(np.eye(9) + train.T @ train / sigma ** 2)
    mean = covariance @ train.T @ y / sigma ** 2
    m_weight = test @ mean
    v_weight = np.einsum('ij,jk,ik->i', test, covariance, test)
    mean_error = float(np.max(np.abs(m_kernel - m_weight)))
    variance_error = float(np.max(np.abs(v_kernel - v_weight)))
    assert mean_error < 1e-11
    assert variance_error < 1e-11
    return dict(
        normalizer_quad_reference=reference,
        normalizer_abs_error=normalizer_error,
        dense_posterior_mean_max_error=mean_error,
        dense_posterior_variance_max_error=variance_error,
    )


def compute():
    started = time.perf_counter()
    normalizer = tanh_second_moment(CONFIG['quadrature_order'])
    checks = independent_checks(normalizer)
    grid = np.linspace(0, 2 * np.pi, CONFIG['grid_points'], endpoint=False)
    data_rng = np.random.default_rng(CONFIG['data_seed'])
    half_angles = np.pi * (np.arange(6) + data_rng.uniform(0.15, 0.85, 6)) / 6
    train_angles = np.concatenate((half_angles, half_angles + np.pi))
    y_true = target(train_angles)
    noise = data_rng.normal(scale=CONFIG['noise_sd'], size=12)
    y = y_true + noise
    x_train, x_grid = circle(train_angles), circle(grid)
    truth = target(grid)

    shape = (len(CONFIG['feature_seeds']), len(CONFIG['bias_variances']))
    posterior_means = np.empty(shape + (len(grid),))
    posterior_variances = np.empty_like(posterior_means)
    spectra = np.empty(shape + (len(grid) // 2 + 1,))
    base_weights, base_biases = [], []
    metrics = []

    for seed_index, seed in enumerate(CONFIG['feature_seeds']):
        rng = np.random.default_rng(seed)
        base_w = rng.standard_normal((CONFIG['width'], 2))
        base_b = rng.standard_normal(CONFIG['width'])
        base_weights.append(base_w)
        base_biases.append(base_b)
        for condition, bias_variance in enumerate(CONFIG['bias_variances']):
            phi_train = feature_matrix(x_train, base_w, base_b, bias_variance, normalizer)
            phi_grid = feature_matrix(x_grid, base_w, base_b, bias_variance, normalizer)
            prediction, variance, k_train = posterior(phi_train, phi_grid, y, CONFIG['noise_sd'])
            train_prediction, _, _ = posterior(phi_train, phi_train, y, CONFIG['noise_sd'])
            spectrum = mode_variances(phi_grid)
            parity_residual = float(np.max(np.abs(phi_grid + np.roll(phi_grid, len(grid) // 2, axis=0))))
            mean_parity = float(np.max(np.abs(prediction + np.roll(prediction, len(grid) // 2))))
            minimum_eigenvalue = float(np.linalg.eigvalsh(k_train).min())
            assert minimum_eigenvalue > -1e-11
            parseval_error = float(abs(spectrum.sum() - np.mean(np.sum(phi_grid ** 2, axis=1))))
            assert parseval_error < 1e-12
            if bias_variance == 0:
                assert parity_residual < 1e-14
                assert mean_parity < 1e-11
                assert float(spectrum[::2].sum()) < 1e-25
            else:
                assert float(spectrum[2::2].sum()) > 1e-4
            posterior_means[seed_index, condition] = prediction
            posterior_variances[seed_index, condition] = variance
            spectra[seed_index, condition] = spectrum
            metrics.append(dict(
                feature_seed=seed, bias_variance=bias_variance,
                grid_rmse=float(np.sqrt(np.mean((prediction - truth) ** 2))),
                training_rmse=float(np.sqrt(np.mean((train_prediction - y) ** 2))),
                grid_target_inside_2posterior_sd_fraction=float(np.mean(np.abs(prediction - truth) <= 2 * np.sqrt(variance))),
                mean_latent_posterior_variance=float(np.mean(variance)),
                mean_prior_variance=float(spectrum.sum()),
                even_mode_variance_excluding_constant=float(spectrum[2::2].sum()),
                constant_mode_variance=float(spectrum[0]),
                feature_antipodal_max_residual=parity_residual,
                posterior_mean_antipodal_max_residual=mean_parity,
                training_kernel_min_eigenvalue=minimum_eigenvalue,
                parseval_abs_error=parseval_error,
            ))

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS / 'results.npz', grid=grid, truth=truth,
        train_angles=train_angles, y=y, y_true=y_true, observation_noise=noise,
        posterior_means=posterior_means, posterior_variances=posterior_variances,
        prior_mode_variances=spectra, base_weights=np.array(base_weights),
        base_biases=np.array(base_biases),
    )
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    metadata = dict(
        config=CONFIG, normalizer=normalizer, checks=checks, metrics=metrics,
        source=dict(
            git_head=head,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            plot_style_sha256=hashlib.sha256((HERE / 'plot_style.py').read_bytes()).hexdigest(),
            command='MPLCONFIGDIR=/private/tmp/disordered-minds-mplconfig OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 _drafts/post-6-quenched-disorder/experiments/inductive_bias.py',
        ),
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
        computed_at_utc=datetime.now(timezone.utc).isoformat(),
        compute_wall_seconds=time.perf_counter() - started,
        interpretation='Deterministic target deliberately violates the unbiased architecture. Band inclusion fractions describe this target, not Bayesian calibration.',
    )
    (RESULTS / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (RESULTS / 'plot_style_snapshot.py').write_bytes((HERE / 'plot_style.py').read_bytes())
    (RESULTS / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(normalizer=normalizer, checks=checks, metrics=metrics), indent=2))


def plot():
    data = np.load(RESULTS / 'results.npz')
    metadata = json.loads((RESULTS / 'results.json').read_text())
    config = metadata['config']
    representative = config['feature_seeds'].index(config['representative_seed'])
    grid = data['grid'] / np.pi
    modes = np.arange(10)
    for theme in ['light', 'dark']:
        colors = configure(theme)
        fig = plt.figure(figsize=(6.8, 6.25), layout='constrained')
        gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.2], hspace=0.12)
        spectrum_ax = fig.add_subplot(gs[0, :])
        spectrum_ax.set_title('Prior variance by angular frequency', loc='left')
        for condition, color, marker, label in [
            (0, colors['ink'], 'o', 'No hidden bias'),
            (1, colors['accent'], 's', 'Hidden bias variance ¼'),
        ]:
            sample = data['prior_mode_variances'][:, condition, :10]
            mean = sample.mean(axis=0)
            spectrum_ax.plot(modes, mean, marker=marker, markersize=4, color=color, label=label)
            # Across-feature-realization range, rather than a fabricated ensemble CI.
            spectrum_ax.fill_between(modes, sample.min(axis=0), sample.max(axis=0), color=color, alpha=0.13, linewidth=0)
        spectrum_ax.set_yscale('symlog', linthresh=1e-5)
        spectrum_ax.set_ylim(-1e-6, 1.4)
        spectrum_ax.set_yticks([0, 1e-4, 1e-2, 1], ['0', r'$10^{-4}$', r'$10^{-2}$', '1'])
        spectrum_ax.set_xticks(modes)
        spectrum_ax.set_xlabel('Fourier mode k')
        spectrum_ax.set_ylabel('Prior variance')
        spectrum_ax.legend(loc='upper right', ncol=2, columnspacing=1.1, handlelength=1.8)
        spectrum_ax.annotate('Even modes absent', xy=(2, 0), xytext=(2.8, 0.025),
                             color=colors['secondary'], fontsize=10.5,
                             arrowprops=dict(arrowstyle='-', color=colors['secondary'], linewidth=0.8))

        for condition, title in [(0, 'No hidden bias'), (1, 'Hidden bias variance ¼')]:
            ax = fig.add_subplot(gs[1, condition])
            mean = data['posterior_means'][representative, condition]
            sd = np.sqrt(data['posterior_variances'][representative, condition])
            ax.fill_between(grid, mean - 2 * sd, mean + 2 * sd, color=colors['accent'], alpha=0.2, linewidth=0)
            ax.plot(grid, data['truth'], color=colors['muted'], linestyle='--', linewidth=1.4, label='Target')
            ax.plot(grid, mean, color=colors['accent'], label='Posterior mean')
            ax.scatter(data['train_angles'] / np.pi, data['y'], color=colors['ink'], s=20, zorder=4, label='Same observations')
            ax.set_title(title, loc='left')
            ax.set_xlim(0, 2)
            ax.set_ylim(-1.6, 1.6)
            ax.set_xticks([0, 1, 2], ['0', 'π', '2π'])
            ax.set_xlabel('Input angle t')
            ax.set_yticks([-1, 0, 1])
            if condition == 0:
                ax.set_ylabel('Function value')
            else:
                ax.tick_params(labelleft=False)
        handles, labels = fig.axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=3, handlelength=1.8, columnspacing=1.2)
        save_figure(fig, 'inductive-bias', theme)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if not args.plot_only:
        compute()
    plot()
