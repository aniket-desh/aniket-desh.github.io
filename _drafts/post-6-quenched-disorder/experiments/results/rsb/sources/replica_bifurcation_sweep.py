#!/usr/bin/env python3
"""Compute and plot a finite-step replica bifurcation; CPU only.

Run from any directory:
  python3 replica_bifurcation.py run
  python3 replica_bifurcation.py check
  python3 replica_bifurcation.py plot

The finite-step saddle is our computation of the one-hidden sine model in
Winer & Hanin (2025), not digitized published data and not finite-network MCMC.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.integrate import quad
from rsb_solver import SineParisi
from plot_style import configure, save_figure
from matplotlib import pyplot as plt

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'/'rsb'
RESULTS.mkdir(parents=True,exist_ok=True)
BETAS=[.5,1,1.5,2,2.5,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.5,6,6.5,7.1,7.5,8]
CONFIG=dict(steps=20,mesh=256,quadrature=128,alpha=1.)

def native(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    return x

def save(name,data):
    (RESULTS/name).write_text(json.dumps(native(data),indent=2)+'\n')

def metadata():
    return dict(date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
                source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['rsb_solver.py','replica_bifurcation.py']},
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
                random_seed=None,initialization='deterministic RS plus .02 total hierarchy spread; break RS at every collapsed continuation point',
                reference='Mike Winer and Boris Hanin, Deep Neural Nets as Hamiltonians, arXiv:2503.23982v2, Appendix A.2; Fig. 2; https://arxiv.org/html/2503.23982v2',
                model='Gaussian x~N(0,I_d); H=sum_i a_i sin(w_i.x); w_ij~N(0,1/d), a_i~N(0,1); N/d=1',
                method='minimize finite-step off-diagonal Parisi profile at fixed diagonal, maximize over diagonal',
                thermodynamic_limit='d,N→∞ with N/d=1; finite 20-step replica ansatz, not a finite-network simulation')

def run():
    solver=SineParisi(**CONFIG)
    rows=[]
    previous=None
    start=time.time()
    for beta in BETAS:
        t=time.time()
        row=solver.solve(beta,initial=previous)
        row['beta']=beta
        row['elapsed_seconds']=time.time()-t
        rows.append(row);previous=row
        save('sweep.json',dict(metadata=metadata(),config=CONFIG,rows=rows,total_seconds=time.time()-start))
        print(f"beta {beta:g}: q={row['q'][0]:.6f}..{row['q'][-1]:.6f}, D={row['D']:.6f}, residual={max(row['profile_residual'],abs(row['diagonal_gradient'])):.2g}, {row['elapsed_seconds']:.1f}s",flush=True)
    plot()

def checks():
    results={}
    # Independent entropy check: for integer n=6, block size 3, the determinant
    # can be calculated directly before analytically taking n->0.
    n,m,c,q0,q1=6,3,.9,.4,.7
    Q=np.full((n,n),q0)
    for j in [0,3]:Q[j:j+3,j:j+3]=q1
    Q+=np.eye(n)*c
    chi=c+m*(q1-q0)
    formula=np.log(c)+np.log(chi/c)/m+np.log(1+n*q0/chi)/n
    results['integer_matrix_entropy_error']=abs(np.linalg.slogdet(Q)[1]/n-formula)
    # Exact linear network: all hierarchical derivatives vanish at c=1,
    # q_i=alpha*beta^2, and logZ/d=alpha*beta^2/2.
    linear=[]
    ls=SineParisi(20,activation='linear')
    for beta in [0,1,4.5,7.1]:
        q=beta**2
        value,g=ls.objective(1+q,np.r_[q,np.zeros(20)],beta)
        linear.append(dict(beta=beta,value=value,value_error=abs(value-.5*beta**2),gradient_max=float(abs(g).max())))
    results['exact_linear_RS']=linear
    # Independent quadrature, avoiding FFTs: terminal scalar Gaussian integral
    # at several field values and readout strengths.
    s=SineParisi(**CONFIG)
    terminal=[]
    for c0,h,a in [(.6,0.,1.),(.9,1.1,-2.),(.6,3.,4.)]:
        beta=7.1
        field=beta*a*torch.sin(s.h)[None,:]
        fft=s.log_convolve(field,torch.tensor(c0)).detach().numpy()[0]
        # Interpolation is unnecessary for mesh-aligned h, use a mesh point.
        index=int(round(h/(2*np.pi)*s.mesh))%s.mesh;h=float(s.h[index])
        offset=abs(beta*a)
        integral,error=quad(lambda z:np.exp(beta*a*np.sin(h+np.sqrt(c0)*z)-offset-.5*z*z)/np.sqrt(2*np.pi),-12,12,epsabs=1e-12,epsrel=1e-11)
        direct=np.log(integral)+offset
        terminal.append(dict(c=c0,h=h,a=a,absolute_error=abs(float(fft[index])-direct)))
    results['terminal_fft_vs_independent_adaptive_quadrature']=terminal
    # Autodiff needs an independent central finite-difference check.
    x=np.r_[1.9,.8,np.full(20,.015)]
    val,g=s.objective(x[0],x[1:],5.)
    gradchecks=[]
    for index in [0,1,5,21]:
        eps=1e-5
        plus=x.copy();minus=x.copy();plus[index]+=eps;minus[index]-=eps
        vp=s.objective(plus[0],plus[1:],5.)[0];vm=s.objective(minus[0],minus[1:],5.)[0]
        fd=(vp-vm)/(2*eps)
        gradchecks.append(dict(index=index,autodiff=g[index],finite_difference=fd,absolute_error=abs(g[index]-fd)))
    results['autodiff_central_difference']=gradchecks
    save('checks.json',dict(metadata=metadata(),checks=results))
    print(json.dumps(native(results),indent=2),flush=True)
    # Refinement points compare spatial mesh, readout quadrature, hierarchy
    # resolution, and a second start. Save every solve, including failures.
    existing=json.loads((RESULTS/'sweep.json').read_text()) if (RESULTS/'sweep.json').exists() else None
    refine=[]
    for beta in [4.5,7.1]:
        for config in [dict(steps=20,mesh=512,quadrature=128,alpha=1.),dict(steps=20,mesh=256,quadrature=256,alpha=1.),dict(steps=40,mesh=256,quadrature=128,alpha=1.)]:
            t=time.time();fit=SineParisi(**config).solve(beta)
            refine.append(dict(config=config,beta=beta,fit=fit,seconds=time.time()-t))
            save('refinements.json',dict(metadata=metadata(),rows=refine))
            print('refinement',config,beta,fit['q'][0],fit['q'][-1],fit['diagonal_gradient'],flush=True)
    return results

def plot():
    data=json.loads((RESULTS/'sweep.json').read_text())
    rows=data['rows'];beta=np.array([r['beta'] for r in rows]);q=np.array([r['q'] for r in rows])
    qrs=np.array([r['rs']['q'] for r in rows])
    for theme in ['light','dark']:
        colors=configure(theme)
        fig,(ax,width)=plt.subplots(2,1,figsize=(6.8,4.7),sharex=True,gridspec_kw=dict(height_ratios=[3.8,1],hspace=.12))
        for i in range(q.shape[1]):
            ax.plot(beta,q[:,i],color=colors['accent'],alpha=.55,lw=.85)
        ax.plot(beta,q[:,0],color=colors['ink'],lw=1.6,label='Smallest / largest overlap')
        ax.plot(beta,q[:,-1],color=colors['ink'],lw=1.6)
        ax.plot(beta,qrs,color=colors['secondary'],lw=1.1,ls=(0,(4,3)),label='Replica-symmetric saddle')
        ax.set_ylabel(r'Replica overlap $q$')
        ax.legend(loc='upper left',fontsize=10)
        ax.text(.98,.04,'One hidden sine layer · $N/d=1$',ha='right',va='bottom',transform=ax.transAxes,fontsize=10,color=colors['secondary'])
        ax.set_ylim(0,max(q[:,-1])*1.08)
        width.plot(beta,q[:,-1]-q[:,0],color=colors['accent'])
        width.axhline(0,lw=.7,color=colors['rule'])
        width.set_ylabel(r'$q_{\max}-q_{\min}$',fontsize=11)
        width.set_xlabel(r'Inverse temperature $\beta$')
        width.set_xlim(.5,8)
        width.set_yticks([0,.4,.8])
        save_figure(fig,'replica-bifurcation',theme)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','check','plot'],nargs='?',default='run')
    args=parser.parse_args()
    {'run':run,'check':checks,'plot':plot}[args.action]()
