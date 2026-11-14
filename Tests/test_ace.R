# Run from the repository root: Rscript --vanilla Tests/test_ace.R
source(file.path('Algorithm', 'ace_fixed_y_classic.R'))
x <- rep(seq(.1, 2, length.out=24), each=2)
for (y in list(sqrt(x), -sqrt(x))) {
    fit <- ace_fixed_y_classic(x, y)
    stopifnot(length(fit$tx)==length(x), all(is.finite(fit$tx)))
    stopifnot(all(diff(fit$tx)>=-1e-12))
    stopifnot(all(fit$tx[seq(1, length(x), by=2)]==fit$tx[seq(2, length(x), by=2)]))
    stopifnot(abs(mean(fit$tx)-mean(x))<1e-12)
    stopifnot(abs(sd(fit$tx)-sd(x))<1e-12)
    stopifnot(max(abs(fit$ty-as.vector(scale(y))))<1e-12)
}
cat('ACE-M checks passed: fixed response, both association directions, order and repeated inputs.\n')
