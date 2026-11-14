"""The unchanged COMET objective with analytic gradient and Hessian products."""
from pathlib import Path
import os, sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
# Dependencies are installed in the active Python environment.
import numpy as np
from scipy.optimize import minimize, isotonic_regression


class Objective:
    def __init__(self,x,y,lam,direction,gap_ratio=.01):
        self.raw=x;self.location=x.mean();self.scale=x.std(ddof=1)
        x=(x-self.location)/self.scale
        self.u,self.g=np.unique(x,return_inverse=True);self.w=np.bincount(self.g).astype(float)
        self.ys=np.bincount(self.g,weights=y-y.mean());self.yn=np.linalg.norm(y-y.mean())
        self.b=self.ys/self.yn;self.n=len(x);self.v=np.sum((x-x.mean())**2)
        self.lam=lam;self.direction=direction;self.gap=gap_ratio*np.min(np.diff(self.u))
        inc=np.diff(self.u)-self.gap
        self.initial=np.r_[self.u[0],np.where(inc>20,inc,np.log(np.expm1(np.minimum(inc,20))))]
        self.cached=None

    def evaluate(self,a):
        if self.cached is not None and np.array_equal(a,self.cached[0]):
            return self.cached[1]
        steps=np.logaddexp(0,a[1:])+self.gap
        sig=np.exp(-np.logaddexp(0,-a[1:]))
        q=np.r_[a[0],a[0]+np.cumsum(steps)];cc=q-np.dot(self.w,q)/self.n
        zn=np.sqrt(np.dot(self.w,cc*cc));r=np.dot(q,self.b)/zn;p=self.w*cc
        delta=q-self.u;f=-self.direction*r+self.lam*np.dot(self.w,delta*delta)/self.v
        dq=-self.direction*(self.b/zn-r*p/zn**2)+2*self.lam*self.w*delta/self.v
        tail=np.cumsum(dq[::-1])[::-1];grad=np.r_[dq.sum(),sig*tail[1:]]
        value=(f,grad,q,sig,zn,r,p,tail);self.cached=(a.copy(),value);return value

    def fun(self,a):
        v=self.evaluate(a);return v[0],v[1]

    def hessp(self,a,v):
        _,_,_,sig,zn,r,p,tail=self.evaluate(a)
        t=np.r_[v[0],v[0]+np.cumsum(sig*v[1:])]
        dt=self.w*t-self.w*np.dot(self.w,t)/self.n
        h=self.direction*((self.b*np.dot(p,t)+p*np.dot(self.b,t))/zn**3+
                          r*dt/zn**2-3*r*p*np.dot(p,t)/zn**4)+2*self.lam*self.w*t/self.v
        answer=np.r_[h.sum(),sig*np.cumsum(h[::-1])[::-1][1:]]
        answer[1:]+=sig*(1-sig)*tail[1:]*v[1:]
        return answer


def fit(x,y,lam=.5,method='trust-krylov',gap_ratio=.01):
    fits=[]
    for direction in [1,-1]:
        objective=Objective(x,y,lam,direction,gap_ratio)
        result=minimize(objective.fun,objective.initial,jac=True,hessp=objective.hessp,
                        method=method,options={'gtol':1e-9,'maxiter':2000})
        f,g,q,*_=objective.evaluate(result.x)
        fits.append((f,result,objective,q,float(np.linalg.norm(g))))
    best=min(fits,key=lambda a:a[0]);z=best[2].location+best[2].scale*best[3][best[2].g]
    assert np.all(np.diff(best[3])>0)
    return z,fits


def fit_projected(x,y,lam=.5,maxiter=10000,tolerance=1e-9,gap_ratio=.01):
    """Solve the same node objective, projecting onto the minimum-gap cone.

    Weighted PAVA is only the feasible-set projection; this is NOT the MOS
    objective. Both Pearson directions and the COMET departure penalty remain.
    """
    fits=[]
    for direction in [1,-1]:
        o=Objective(x,y,lam,direction,gap_ratio);w=o.w;gap=o.gap*np.arange(len(o.u));q=o.u.copy()
        alpha=o.v/(2*lam)
        def evaluate(q):
            cc=q-np.dot(w,q)/o.n;zn=np.sqrt(np.dot(w,cc*cc));r=np.dot(q,o.b)/zn
            d=q-o.u;f=-direction*r+lam*np.dot(w,d*d)/o.v
            g=-direction*(o.b/zn-r*w*cc/zn**2)+2*lam*w*d/o.v
            return f,g
        for iteration in range(maxiter):
            f,g=evaluate(q)
            while True:
                candidate=isotonic_regression(q-alpha*g/w-gap,weights=w).x+gap
                delta=candidate-q;newf,_=evaluate(candidate)
                bound=f+np.dot(g,delta)+np.dot(w,delta*delta)/(2*alpha)
                if newf<=bound+1e-15 or alpha<1e-12:break
                alpha*=.5
            q=candidate
            # Certify the returned vector at a fixed natural step so that the
            # stopping residual can be reconstructed from saved node scores.
            _,final_gradient=evaluate(q);probe_alpha=o.v/(2*lam)
            probe=isotonic_regression(q-probe_alpha*final_gradient/w-gap,weights=w).x+gap
            residual=np.max(abs(probe-q))/probe_alpha
            if residual<=tolerance:break
            alpha=min(o.v/(2*lam),alpha*1.1)
        f,g=evaluate(q)
        fits.append(dict(loss=f,iterations=iteration+1,residual=residual,success=residual<=tolerance,
                         direction=direction,objective=o,q=q))
    best=min(fits,key=lambda x:x['loss']);o=best['objective'];z=o.location+o.scale*best['q'][o.g]
    assert np.all(np.diff(best['q'])>0)
    return z,fits


