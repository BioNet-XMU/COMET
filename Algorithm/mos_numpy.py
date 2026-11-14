"""Paper MOS implementation; constant fitted scores remain constant."""
import numpy as np
from scipy.optimize import isotonic_regression

def mos(x,y):
    u,inverse=np.unique(x,return_inverse=True);w=np.bincount(inverse).astype(float)
    if len(u)==1 or np.std(x,ddof=1)<=np.spacing(max(1,abs(x.mean()))):return x.copy()
    average=np.bincount(inverse,weights=y)/w
    inc=isotonic_regression(average,weights=w).x;negative=isotonic_regression(-average,weights=w).x
    se1=np.sum((y-inc[inverse])**2);se2=np.sum((y+negative[inverse])**2);tol=1e-12*max(1,min(se1,se2))
    choose_increasing=se1<se2-tol if abs(se1-se2)>tol else np.corrcoef(x,y)[0,1]>=0
    score=(inc if choose_increasing else negative)[inverse]
    if np.std(score,ddof=1)<=1e-12:return np.full_like(x,x.mean())
    return x.mean()+x.std(ddof=1)*(score-score.mean())/score.std(ddof=1)
