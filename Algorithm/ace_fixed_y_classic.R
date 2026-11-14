# Fixed-response, single-predictor monotone ACE smoother.
# Smoothing/monotone/plateau steps follow smothr(l=3), acepack 1.4.1 avas.f.
# The response is fixed up to centering/scaling; no nonlinear Y update occurs.
# stats::supsmu supplies Friedman's variable-span conditional expectation;
# stats::isoreg supplies the equal-weight monotone projection (montne).
# Adapted algorithm copyright 1985,2007 Jerome H. Friedman.
# Permission to use, copy, modify, distribute, and sell this software and
# its documentation for any purpose is hereby granted without fee,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation. No representations are made about the
# suitability of this software for any purpose. It is provided "as is"
# without express or implied warranty.
ace_fixed_y_classic <- function(x,y) {
  stopifnot(length(x)==length(y), all(is.finite(x)), all(is.finite(y)))
  if(sd(x)==0 || sd(y)==0)stop('Constant feature/response in ACE validation.')
  order_x<-order(x);xs<-x[order_x];ys<-as.vector(scale(y))[order_x]
  smooth<-stats::supsmu(xs,ys,span='cv',periodic=FALSE,bass=0)
  # supsmu returns one value per DISTINCT input, not per observation.
  # Restore repeated observations before the equal-weight projection.
  sm<-smooth$y[match(xs,smooth$x)]
  stopifnot(length(sm)==length(xs),all(is.finite(sm)))
  inc<-stats::isoreg(seq_along(sm),sm)$yf
  dec<-rev(stats::isoreg(seq_along(sm),rev(sm))$yf)
  # Choose the closer monotone approximation to the conditional smoother,
  # not the isotonic fit to the raw response (the latter is MOS).
  if(sum((sm-inc)^2)<sum((sm-dec)^2))sm<-inc else sm<-dec
  # isoreg can return adjacent mathematical plateaus separated only by roundoff.
  # Merge these before spreading, so an interior zero gap is not mistaken for
  # the boundary case in the original plateau rule.
  tolerance<-64*.Machine$double.eps*max(1,max(abs(sm)))
  blocks<-cumsum(c(TRUE,abs(diff(sm))>tolerance))
  before<-sm;sm<-as.vector(ave(sm,blocks,FUN=mean))
  stopifnot(max(abs(sm-before))<1e-10*max(1,max(abs(sm))))
  level_repairs<-sum(sm!=before)
  j<-1L;n<-length(sm)
  while(j<=n) {
    j0<-j
    while(j<n && sm[j+1]==sm[j])j<-j+1L
    if(j>j0) {
      a<-if(j0>1) .5*(sm[j0]-sm[j0-1]) else 0
      b<-if(j<n) .5*(sm[j+1]-sm[j]) else 0
      delta<-(a+b)/(j-j0)
      if(a==0 || b==0)delta<-2*delta
      if(a==0)a<-b
      sm[j0:j]<-sm[j0:j]-a+delta*((j0:j)-j0)
    }
    j<-j+1L
  }
  # Original tied inputs remain tied, as required by a function of X.
  # Singleton groups are already their own mean; pool only actual tied inputs.
  tied<-duplicated(xs)|duplicated(xs,fromLast=TRUE)
  for(value in unique(xs[tied]))sm[xs==value]<-mean(sm[xs==value])
  fallback<-FALSE
  if(sd(sm)==0) {
    return(list(tx=x,ty=as.vector(scale(y)),fallback=TRUE,roundoff_repairs=level_repairs))
  }
  if(cor(sm,xs,method='spearman')<0)sm<--sm
  if(min(diff(sm)) < -1e-12*max(1,max(abs(sm))))stop(sprintf('Classic monotone smoother failed: gap %.17g, range %.17g.',min(diff(sm)),diff(range(sm))))
  repairs<-sum(diff(sm)<0)+level_repairs
  sm<-cummax(sm) # Remove only verified floating-point reversals of monotone scores.
  z<-numeric(n);z[order_x]<-mean(x)+sd(x)*(sm-mean(sm))/sd(sm)
  list(tx=z,ty=as.vector(scale(y)),fallback=fallback,roundoff_repairs=repairs)
}
