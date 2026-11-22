"""Finite-step Parisi functional for a Gaussian-input, one-hidden sine network.

H(x)=sum_i a_i sin(w_i.x), w_ij~N(0,1/d), a_i~N(0,1), N/d=alpha.
For n replicas the hidden-site factor is
    E_a E_{z~N(0,Q)} exp(beta*a*sum_a sin(z_a)).
Appendix A.2 of Winer--Hanin gives its hierarchical Gaussian recursion. Since
sin is periodic, Gaussian convolution is exp(-variance*k^2/2) in Fourier space.
The readout a is quenched by taking E_a of the n->0 log recursion, not log E_a.

Q has diagonal D and off-diagonal levels q_0<=...<=q_K, with breakpoints m_j.
chi_i = D-q_K + sum_{j>i} m_j*(q_j-q_{j-1}). Then
    lim logdet(Q)/n = log(c)+q_0/chi_0
                       +sum_j log(chi_{j-1}/chi_j)/m_j.
The functional is half*(logdet/n-D+1)+alpha*site. Minimize the off-diagonal
profile at fixed D; maximize over D. This is a replica ansatz, not a theorem
that this network obeys the Parisi variational principle.
"""
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, root, minimize_scalar, least_squares

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

class SineParisi:
    def __init__(self, steps=10, mesh=256, quadrature=64, alpha=1., activation="sin"):
        self.steps, self.mesh, self.quadrature, self.alpha=steps,mesh,quadrature,alpha
        self.activation=activation
        self.m=torch.arange(1,steps+1)/(steps+1) if steps else torch.empty(0)
        self.h=torch.arange(mesh)*2*np.pi/mesh
        self.k=torch.arange(mesh//2+1)
        a,w=hermgauss(quadrature)
        self.a=torch.tensor(a*np.sqrt(2))
        self.weights=torch.tensor(w/np.sqrt(np.pi))
        self.clips=0
    def convolve(self, values, variance):
        result=torch.fft.irfft(torch.fft.rfft(values)*torch.exp(-.5*variance*self.k**2),n=self.mesh)
        return result
    def log_convolve(self, f, variance, power=1.):
        scaled=power*f
        offset=torch.amax(scaled,dim=-1,keepdim=True)
        result=self.convolve(torch.exp(scaled-offset),variance)
        # Only Fourier roundoff at negligibly weighted quadrature tails can
        # trigger this floor; check refinement and weighted clipped mass.
        return (torch.log(torch.clamp(result,min=1e-15))+offset)/power
    def value(self, D, increments, beta):
        q=torch.cumsum(increments,0)
        c=D-q[-1]
        if self.steps:
            weighted=self.m*increments[1:]
            chi=torch.cat((c+torch.flip(torch.cumsum(torch.flip(weighted,[0]),0),[0]),c.reshape(1)))
            logdet=torch.log(c)+q[0]/chi[0]+torch.sum(torch.log(chi[:-1]/chi[1:])/self.m)
        else:
            logdet=torch.log(c)+q[0]/c
        if self.activation == "linear":
            # Gaussian integrals are analytic, at every hierarchy level.
            site=.5*beta**2*(c+torch.sum(self.m*increments[1:]))
        else:
            f=self.log_convolve(beta*self.a[:,None]*torch.sin(self.h),c)
            for j in range(self.steps-1,-1,-1):
                f=self.log_convolve(f,increments[j+1],self.m[j])
            site=torch.sum(self.weights*self.convolve(f,q[0])[:,0])
        return .5*(logdet-D+1)+self.alpha*site
    def objective(self,D,increments,beta):
        var=torch.tensor(np.r_[D,increments],requires_grad=True)
        val=self.value(var[0],var[1:],beta)
        grad=torch.autograd.grad(val,var)[0]
        return float(val.detach()),grad.detach().numpy()
    def solve_rs(self,beta,guess=None):
        if self.activation == "linear":
            q=self.alpha*beta**2
            return dict(D=1.+q,q=q,c=1.,residual=0.,success=True,message="Exact Gaussian linear solution")
        rs=SineParisi(0,self.mesh,self.quadrature,self.alpha)
        if guess is None: guess=[1.+.22*beta+.18*beta, max(.18*beta,1e-5)]
        # Root on c and q. Log parameters keep the Gaussian variances positive.
        def fun(v):
            c,q=np.exp(v)
            _,g=rs.objective(c+q,[q],beta)
            return np.array([g[0],g[0]+g[1]])
        fit=least_squares(fun,np.log([max(guess[0]-guess[1],.05),max(guess[1],1e-7)]),xtol=1e-12,gtol=1e-12,ftol=1e-12,max_nfev=100)
        c,q=np.exp(fit.x)
        _,g=rs.objective(c+q,[q],beta)
        return dict(D=float(c+q),q=float(q),c=float(c),residual=float(np.max(np.abs(g))),success=bool(fit.success),message=str(fit.message))
    def minimize_profile(self,D,beta,initial=None,maxiter=400,ftol=2e-12):
        if initial is None:
            initial=np.r_[.2,np.full(self.steps,.001)]
        initial=np.maximum(np.asarray(initial),0)
        if initial.sum()>=D-.01: initial*=max(D-.02,.01)/initial.sum()
        cache={}
        def fg(x):
            key=tuple(x)
            if key not in cache: cache[key]=self.objective(D,x,beta)
            val,g=cache[key]
            return val,g[1:]
        result=minimize(fg,initial,jac=True,method='SLSQP',bounds=[(0,None)]*(self.steps+1),
            constraints=[{'type':'ineq','fun':lambda x:D-.005-x.sum(),'jac':lambda x:-np.ones_like(x)}],
            options={'maxiter':maxiter,'ftol':ftol})
        val,g=self.objective(D,result.x,beta)
        active=result.x>1e-6
        kkt=np.where(active,g[1:],np.minimum(g[1:],0))
        return dict(D=float(D),increments=result.x, q=np.cumsum(result.x),c=float(D-result.x.sum()),
                    value=val,diagonal_gradient=float(g[0]),profile_residual=float(np.max(np.abs(kkt))),
                    success=bool(result.success),iterations=int(result.nit),message=str(result.message))
    def solve(self,beta,initial=None,D0=None,verbose=False):
        rs=self.solve_rs(beta,guess=None if initial is None else [initial['D'],initial['q'][0]])
        if D0 is None: D0=rs['D']
        if initial is None:
            # A real small hierarchical perturbation, not an exactly RS seed.
            inc=np.r_[rs['q']-.01,np.full(self.steps,.02/max(self.steps,1))]
        else:
            inc=np.r_[initial["q"][0],np.diff(initial["q"])]
            # A collapsed previous saddle must be perturbed again: otherwise
            # exact RS symmetry prevents a local solver from finding RSB.
            if self.steps and inc[1:].sum()<.01:
                inc[0]=max(inc[0]-.01,0)
                inc[1:]+=.02/self.steps
        inc=np.maximum(inc,0)
        profile=inc
        evals=[]
        def outer(D):
            nonlocal profile
            fit=self.minimize_profile(D,beta,profile)
            profile=fit['increments']
            evals.append(fit)
            if verbose: print('D',D,'F',fit['value'],'gd',fit['diagonal_gradient'],'dq',fit['q'][-1]-fit['q'][0],flush=True)
            return -fit['value']
        # Self-overlap is the maximization direction; the inner profile is the
        # minimization direction. A bounded scalar search avoids saddle drift.
        result=minimize_scalar(outer,bounds=(max(.3,D0*.55),max(D0*1.6,D0+.8)),method='bounded',
                               options={'xatol':2e-6,'maxiter':35})
        fit=self.minimize_profile(result.x,beta,profile,maxiter=1000,ftol=2e-14)
        fit['rs']=rs
        fit['outer_evaluations']=len(evals)
        fit['breakpoints']=self.m.numpy()
        return fit
