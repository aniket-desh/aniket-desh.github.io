"""Finite-width joint output laws versus their fixed-probe ReLU NNGP.

All hidden and readout weights are redrawn for every network realization.
Run with --compute, --plot, --animation, or no flags for all three.
The animation couples widths through prefixes of one neuron sequence per network.
No network is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis

from plot_style import ROOT, configure, save_figure

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results' / 'nngp'


def relu_kernel(probes):
    """Unit-variance kernel for a_i~N(0,2), w_i~N(0,I_2)."""
    rho = np.clip(probes @ probes.T, -1.0, 1.0)
    angle = np.arccos(rho)
    return (np.sqrt(np.maximum(0.0, 1.0 - rho**2)) + (np.pi - angle) * rho) / np.pi


def network_outputs(rng, probes, width, samples, chunk=128):
    outputs = np.empty((samples, len(probes)))
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        hidden = rng.normal(size=(size, width, probes.shape[1]))
        readout = np.sqrt(2.0) * rng.normal(size=(size, width))
        preactivation = np.einsum('bnd,pd->bnp', hidden, probes)
        outputs[start:start+size] = np.einsum(
            'bn,bnp->bp', readout, np.maximum(preactivation, 0.0)
        ) / np.sqrt(width)
    return outputs


def coupled_network_outputs(rng, probes, widths, samples, chunk=128):
    """Each point keeps its neuron sequence as N increases; no interpolation."""
    widths = np.asarray(widths, dtype=int)
    outputs = np.empty((len(widths), samples, len(probes)))
    for start in range(0, samples, chunk):
        size = min(chunk, samples-start)
        hidden = rng.normal(size=(size, int(widths.max()), probes.shape[1]))
        readout = np.sqrt(2.0) * rng.normal(size=(size, int(widths.max())))
        preactivation = np.einsum('bnd,pd->bnp', hidden, probes)
        prefix = np.cumsum(readout[:, :, None] * np.maximum(preactivation, 0.0), axis=1)
        outputs[:, start:start+size] = np.transpose(prefix[:, widths-1], (1,0,2)) / np.sqrt(widths)[:,None,None]
    return outputs


def mmd2_unbiased(x, y, bandwidth_squared):
    """Unbiased RBF MMD² U-statistic; signed finite-sample values are retained."""
    n, m = len(x), len(y)
    kxx = np.exp(-cdist(x, x, 'sqeuclidean') / (2 * bandwidth_squared))
    kyy = np.exp(-cdist(y, y, 'sqeuclidean') / (2 * bandwidth_squared))
    kxy = np.exp(-cdist(x, y, 'sqeuclidean') / (2 * bandwidth_squared))
    return float((kxx.sum()-n)/(n*(n-1)) + (kyy.sum()-m)/(m*(m-1)) - 2*kxy.mean())


def fixed_probes(seed, count=8):
    rng = np.random.default_rng(seed)
    # Select geometric conditioning only, before seeing any network outputs.
    # Angular gaps and distances from antipodality are bounded away from zero.
    for attempt in range(10000):
        angles = np.sort(rng.uniform(-np.pi, np.pi, count))
        separation = np.abs(np.angle(np.exp(1j*(angles[:, None]-angles[None, :]))))
        pairs = separation[np.triu_indices(count, 1)]
        if pairs.min() > 0.28 and np.min(np.abs(pairs-np.pi)) > 0.10:
            return angles, np.column_stack((np.cos(angles), np.sin(angles))), attempt+1
    raise RuntimeError('Could not generate separated non-antipodal probes')


def independent_checks(probes, kernel):
    # Independent angular quadrature using E[r²]=2 for a 2D Gaussian weight.
    theta = (np.arange(32768)+0.5) * (2*np.pi/32768)
    features = np.maximum(np.column_stack((np.cos(theta), np.sin(theta))) @ probes.T, 0)
    angular_kernel = 4 * features.T @ features / len(theta)
    quadrature_error = float(np.max(np.abs(angular_kernel-kernel)))
    assert quadrature_error < 1e-7, quadrature_error
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.linalg.eigvalsh(kernel).min() > 0

    # Tiny independent double-sum reference protects diagonal normalization.
    x = np.array([[0., 0.], [1., -1.], [2., 0.5]])
    y = np.array([[-0.5, 0.], [1.5, 2.], [0., -2.], [2., 2.]])
    bw2 = 2.3
    rbf = lambda a, b: np.exp(-np.sum((a-b)**2)/(2*bw2))
    reference = sum(rbf(a,b) for i,a in enumerate(x) for j,b in enumerate(x) if i!=j)/(len(x)*(len(x)-1))
    reference += sum(rbf(a,b) for i,a in enumerate(y) for j,b in enumerate(y) if i!=j)/(len(y)*(len(y)-1))
    reference -= 2*sum(rbf(a,b) for a in x for b in y)/(len(x)*len(y))
    mmd_error = abs(mmd2_unbiased(x,y,bw2)-reference)
    assert mmd_error < 1e-13
    return dict(kernel_angular_quadrature_max_error=quadrature_error,
                mmd_tiny_reference_error=float(mmd_error))


def compute():
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    widths = np.array([1, 2, 4, 8, 32, 128, 512])
    samples, batches, metric_samples, gp_groups = 2000, 8, 1024, 32
    angles, probes, attempts = fixed_probes(2026112102)
    kernel = relu_kernel(probes)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    gp_root = eigenvectors * np.sqrt(eigenvalues)[None, :]
    whitener = eigenvectors / np.sqrt(eigenvalues)[None, :]
    bandwidth_squared = 2.0 * len(probes)
    checks = independent_checks(probes, kernel)
    scatter_outputs = np.empty((len(widths), samples, len(probes)))
    metric_outputs = np.empty((len(widths), batches, metric_samples, len(probes)))
    metric_gp_outputs = np.empty_like(metric_outputs)
    mmd = np.empty((len(widths), batches))
    covariance_z = []
    seeds = {'scatter': [], 'network_metric': [], 'gp_metric': [], 'gp_reference': []}

    for wi, width in enumerate(widths):
        seed = 220000 + int(width)
        seeds['scatter'].append(seed)
        outputs = network_outputs(np.random.default_rng(seed), probes, int(width), samples)
        scatter_outputs[wi] = outputs
        products = outputs[:, :, None] * outputs[:, None, :]
        product_se = products.std(axis=0, ddof=1) / np.sqrt(samples)
        zscore = np.max(np.abs(products.mean(axis=0)-kernel)/product_se)
        covariance_z.append(float(zscore))
        assert zscore < 7.0, (int(width), zscore)
        for batch in range(batches):
            seed_net = 330000 + 1000*wi + batch
            seed_gp = 440000 + 1000*wi + batch
            seeds['network_metric'].append(seed_net)
            seeds['gp_metric'].append(seed_gp)
            net = network_outputs(np.random.default_rng(seed_net), probes, int(width), metric_samples)
            gp = np.random.default_rng(seed_gp).normal(size=(metric_samples, len(probes))) @ gp_root.T
            metric_outputs[wi, batch] = net
            metric_gp_outputs[wi, batch] = gp
            mmd[wi, batch] = mmd2_unbiased(net @ whitener, gp @ whitener, bandwidth_squared)
        print(f'width={width}: mean unbiased MMD²={mmd[wi].mean():.6g}', flush=True)

    # Each independent null group uses the same mean-of-eight statistic as a
    # plotted network point, with the same 1,024 samples per distribution/batch.
    gp_reference_batches = np.empty((gp_groups, batches))
    gp_reference_outputs = np.empty((gp_groups, batches, 2, metric_samples, len(probes)))
    for group in range(gp_groups):
        group_seeds = []
        for batch in range(batches):
            seed = 660000 + 1000*group + batch
            group_seeds.append(seed)
            rng = np.random.default_rng(seed)
            pair_outputs = rng.normal(size=(2, metric_samples, len(probes))) @ gp_root.T
            gp_reference_outputs[group, batch] = pair_outputs
            gp_reference_batches[group, batch] = mmd2_unbiased(
                pair_outputs[0] @ whitener, pair_outputs[1] @ whitener, bandwidth_squared)
        seeds['gp_reference'].append(group_seeds)
    gp_reference = gp_reference_batches.mean(axis=1)
    # A zero-expectation reference must be compatible with its empirical MC error.
    baseline_se = gp_reference.std(ddof=1)/np.sqrt(gp_groups)
    assert abs(gp_reference.mean()) < 4*baseline_se
    # The exact one-layer marginal fourth cumulant is an independent CLT benchmark.
    marginal_kurtosis = kurtosis(metric_outputs.reshape(len(widths), -1, len(probes)),
                                axis=1, fisher=True, bias=False)
    checks.update(kernel_min_eigenvalue=float(eigenvalues.min()),
                  kernel_condition_number=float(eigenvalues.max()/eigenvalues.min()),
                  covariance_max_standard_error_units_by_width=covariance_z,
                  gp_reference_mean_over_se=float(gp_reference.mean()/baseline_se))
    metrics = dict(widths=widths.tolist(),
                   mmd_mean=mmd.mean(axis=1).tolist(),
                   mmd_batch_standard_deviation=mmd.std(axis=1, ddof=1).tolist(),
                   mmd_mean_standard_error=(mmd.std(axis=1, ddof=1)/np.sqrt(batches)).tolist(),
                   gp_reference_mean=float(gp_reference.mean()),
                   gp_reference_central_90_percent=np.quantile(gp_reference,[.05,.95]).tolist(),
                   marginal_excess_kurtosis_by_probe=marginal_kurtosis.tolist(),
                   theoretical_marginal_excess_kurtosis=(15/widths).tolist(),
                   checks=checks, elapsed_seconds=time.perf_counter()-started)
    config = dict(seed_probes=2026112102, probe_selection_attempts=attempts,
                  widths=widths.tolist(), display_networks_per_width=samples,
                  metric_batches=batches, networks_per_metric_batch=metric_samples,
                  independent_gp_reference_groups=gp_groups,
                  gp_reference_batches_per_group=batches,
                  gp_reference_statistic='mean of eight independent 1024-versus-1024 GP-vs-GP MMD estimates',
                  seed_lists=seeds,
                  hidden_weight_covariance='I_2', readout_variance=2.0,
                  output='sum_i a_i ReLU(w_i dot x) / sqrt(N)',
                  metric='unbiased RBF MMD squared after whitening with theoretical GP covariance',
                  bandwidth_squared=bandwidth_squared,
                  extra_widths_reason='Widths 1,2,4 resolve approach before the finite-sample GP reference floor.',
                  source_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plot_helper_sha256=hashlib.sha256((HERE/'plot_style.py').read_bytes()).hexdigest(),
                  source_snapshots=['source/nngp_correspondence.py','source/plot_style.py'],
                  environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    np.savez_compressed(RESULTS/'data.npz', widths=widths, angles=angles, probes=probes,
                        kernel=kernel, whitener=whitener, gp_root=gp_root,
                        scatter_outputs=scatter_outputs, metric_outputs=metric_outputs,
                        metric_gp_outputs=metric_gp_outputs, mmd=mmd,
                        gp_reference=gp_reference, gp_reference_batches=gp_reference_batches,
                        gp_reference_outputs=gp_reference_outputs,
                        marginal_kurtosis=marginal_kurtosis)
    (RESULTS/'source').mkdir(exist_ok=True)
    for source in [Path(__file__), HERE/'plot_style.py']:
        (RESULTS/'source'/source.name).write_bytes(source.read_bytes())
    (RESULTS/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (RESULTS/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)


def plot():
    data = np.load(RESULTS/'data.npz')
    widths = data['widths']
    kernel = data['kernel']
    # Pair chosen geometrically in advance: probes 0 and 2, not by output appearance.
    pair = [0, 2]
    pair_cov = kernel[np.ix_(pair,pair)]
    vals, vecs = np.linalg.eigh(pair_cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:,order]
    angle = np.degrees(np.arctan2(vecs[1,0],vecs[0,0]))
    # The views show the same central window; report out-of-window realizations.
    view = 4.5
    outside = {}
    for theme in ['light','dark']:
        colors = configure(theme)
        fig, axes = plt.subplots(1,2,figsize=(6.8,3.45),layout='constrained')
        for col, width in enumerate([2,512]):
            ax = axes[col]
            wi = int(np.flatnonzero(widths==width)[0])
            xy = data['scatter_outputs'][wi][:,pair]
            outside_count = int(np.sum(np.any(np.abs(xy)>view,axis=1)))
            outside[str(width)] = outside_count
            ax.scatter(xy[:,0],xy[:,1],s=4,alpha=.14,color=colors['accent'],edgecolors='none',rasterized=True)
            ellipse = Ellipse((0,0),2*np.sqrt(5.991*vals[0]),2*np.sqrt(5.991*vals[1]),
                              angle=angle,fill=False,edgecolor=colors['ink'],linewidth=1.4,
                              linestyle='--',label='GP 95% ellipse')
            ax.add_patch(ellipse)
            ax.set(xlim=(-view,view),ylim=(-view,view),aspect='equal',
                   xlabel=r'$f(x^1)$',ylabel=r'$f(x^3)$',title=f'Width {width}')
            ax.set_xticks([-4,0,4]);ax.set_yticks([-4,0,4])
            ax.text(.03,.97,f'{outside_count} / {len(xy)} outside view',transform=ax.transAxes,
                    ha='left',va='top',fontsize=9.5,color=colors['secondary'])
            if col==0: ax.legend(loc='lower right',fontsize=9)

        save_figure(fig,'nngp-correspondence',theme)
    (RESULTS/'plot_metadata.json').write_text(json.dumps(dict(
        pair_probe_indices=pair,scatter_axis_window=[-view,view],
        scatter_outside_view_counts=outside,display_networks=2000,
        layout='only the two paired-output scatter panels; MMD results retained offline',
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plot_helper_sha256=hashlib.sha256((HERE/'plot_style.py').read_bytes()).hexdigest(),
    ),indent=2)+'\n')


def animation():
    """Export true finite-width frames, retaining point identity across widths."""
    started = time.perf_counter()
    data = np.load(RESULTS/'data.npz')
    pair = np.array([0, 2])
    probes = data['probes'][pair]
    covariance = data['kernel'][np.ix_(pair,pair)]
    widths = np.array(list(range(1,17)) + [20,24,32,48,64,96,128,192,256,384,512,768,1024])
    seed, samples, chunk, view = 7701024, 2000, 128, 4.5
    outputs = coupled_network_outputs(np.random.default_rng(seed),probes,widths,samples,chunk)
    assert np.isfinite(outputs).all()

    # Independent explicit-neuron reference checks prefix selection and sqrt(N).
    tiny_widths = np.array([1,2,7])
    actual = coupled_network_outputs(np.random.default_rng(99007),probes,tiny_widths,3)
    rng = np.random.default_rng(99007)
    hidden = rng.normal(size=(3,7,2))
    readout = np.sqrt(2.0) * rng.normal(size=(3,7))
    exact = np.zeros_like(actual)
    for wi, width in enumerate(tiny_widths):
        for b in range(3):
            for p, probe in enumerate(probes):
                exact[wi,b,p] = sum(readout[b,i]*max(float(hidden[b,i] @ probe),0)
                                     for i in range(width)) / np.sqrt(width)
    prefix_error = float(np.max(np.abs(actual-exact)))
    assert prefix_error < 1e-13

    products = outputs[:,:,:,None] * outputs[:,:,None,:]
    product_se = products.std(axis=1,ddof=1)/np.sqrt(samples)
    covariance_z = np.max(np.abs(products.mean(axis=1)-covariance)/product_se,axis=(1,2))
    assert covariance_z.max() < 7.0
    outside = np.sum(np.any(np.abs(outputs)>view,axis=2),axis=1)
    rounded = np.round(outputs,5)
    rounding_error = float(np.max(np.abs(outputs-rounded)))
    assert rounding_error <= 5.000001e-6
    eigenvalues,eigenvectors = np.linalg.eigh(covariance)
    phase = np.linspace(0,2*np.pi,129)
    circle = np.column_stack((np.cos(phase),np.sin(phase)))
    ellipse = circle @ (eigenvectors*np.sqrt(-2*np.log(.05)*eigenvalues)).T
    payload = dict(schema_version=1, seed=seed, samples=samples,
                   model=dict(hidden_weight_covariance='I_2',readout_variance=2,
                              output='sum_i a_i ReLU(w_i dot x) / sqrt(N)',
                              coupling='same neuron sequence at every width; take its first N terms'),
                   probe_indices=pair.tolist(),probes=probes.tolist(),axis_limits=[-view,view],
                   gp_covariance=covariance.tolist(),
                   ellipse=dict(probability=.95,points=np.round(ellipse,7).tolist()),
                   frames=[dict(width=int(width),outside_view=int(outside[wi]),points=rounded[wi].tolist())
                           for wi,width in enumerate(widths)])
    asset = ROOT/'assets/disordered-minds/nngp-width-animation.json'
    encoded = json.dumps(payload,separators=(',',':'),allow_nan=False).encode()
    assert len(encoded) <= 2_000_000, len(encoded)
    asset.write_bytes(encoded)
    np.savez_compressed(RESULTS/'animation_data.npz',widths=widths,probes=probes,
                        covariance=covariance,outputs=outputs,outside_view=outside)
    config = dict(seed=seed,samples=samples,chunk=chunk,widths=widths.tolist(),
                  probe_indices=pair.tolist(),coupling=payload['model']['coupling'],
                  axis_limits=payload['axis_limits'],json_decimals=5,
                  json_bytes=len(encoded),json_sha256=hashlib.sha256(encoded).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plot_helper_sha256=hashlib.sha256((HERE/'plot_style.py').read_bytes()).hexdigest(),
                  source_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  checks=dict(explicit_prefix_reference_max_error=prefix_error,
                              maximum_rounding_error=rounding_error,
                              covariance_max_standard_error_units_by_width=covariance_z.tolist()),
                  elapsed_seconds=time.perf_counter()-started)
    (RESULTS/'animation_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (RESULTS/'source').mkdir(exist_ok=True)
    for source in [Path(__file__),HERE/'plot_style.py']:
        (RESULTS/'source'/source.name).write_bytes(source.read_bytes())
    print(f'Animation: {len(widths)} actual finite-width frames; {len(encoded):,} JSON bytes',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compute',action='store_true')
    parser.add_argument('--plot',action='store_true')
    parser.add_argument('--animation',action='store_true')
    args=parser.parse_args()
    all_steps = not (args.compute or args.plot or args.animation)
    if args.compute or all_steps: compute()
    if args.plot or all_steps: plot()
    if args.animation or all_steps: animation()
