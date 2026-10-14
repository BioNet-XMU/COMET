# COMET

**Correlation Optimization via Monotonicity-Enforced Transformation**

COMET is a phenotype-guided transformation for metabolomics data. It adjusts the spacing between metabolite abundances to strengthen their linear association with a continuous phenotype while preserving abundance ranks across samples. A displacement penalty limits deviation from the input values.

A separate strictly increasing transformation is fitted to each feature. Distinct input abundances remain distinct at the fitted nodes, and existing ties are retained. The phenotype is not transformed. The fitted node pairs can be retained and applied to new samples without using their phenotypes.

Repository: https://github.com/BioNet-XMU/COMET

## Files

| File | Purpose |
| --- | --- |
| `transform_COMET_continuous_stable.m` | Numerically stable implementation used in the manuscript's pregnancy-data and penalty-sensitivity analyses. Recommended for new applications. |
| `transform_COMET_continuous.m` | Original implementation used in the manuscript's simulation experiments; retained for reproducibility. |
| `example_COMET.m` | Self-contained synthetic example: fit, inspect ranks, and apply fitted maps to new samples. No external data files are required. |
| `PROVENANCE.md` | Source paths and checksums for the unchanged algorithm files. |

Both implementations use the same objective. The stable version additionally guards against positive knot increments rounding to zero in floating-point arithmetic. They have different diagnostic fields and need not produce bitwise-identical results.

This directory contains the core algorithm and a usage example. It does not contain the complete manuscript benchmark suite or public datasets.

## Requirements

The manuscript analyses used **MATLAB R2024b**, with:

- **Optimization Toolbox**: `fminunc` and `optimoptions`.
- **Statistics and Machine Learning Toolbox**: `corr` for Pearson and Spearman diagnostics.

The core functions do not require Parallel Computing Toolbox. Helper functions are included as local functions in the two `.m` files. Earlier MATLAB versions have not been verified for this deposit.

## Quick start

Download or clone this repository, set the MATLAB current folder to it, and run:

```matlab
example_COMET
```

To transform your own training data:

```matlab
% Xtrain: n-by-p numeric matrix (samples in rows, features in columns).
% ytrain: n-by-1 continuous phenotype, aligned with the rows of Xtrain.
[Ztrain, model] = transform_COMET_continuous_stable( ...
    Xtrain, ytrain, 0.5, 0.01, 400, 10000, 1e-9);
```

**Pass all seven arguments.** Although the original source comments mention defaults, neither implementation supplies omitted-argument defaults.

### Inputs

| Argument | Meaning | Setting used in the main manuscript comparisons |
| --- | --- | --- |
| `matX` | n-by-p abundance matrix | User-provided |
| `cvY` | Continuous phenotype with n observations | User-provided |
| `dbLamda` | Nonnegative displacement-penalty weight, lambda | `0.5` |
| `dbMinGapRatio` | Minimum transformed-node gap relative to the smallest distinct input gap; use a value strictly between 0 and 1 | `0.01` |
| `nMaxIteration` | Maximum iterations for each optimization direction | `400` |
| `maxFunctionEvaluations` | Maximum objective evaluations for each direction | `10000` |
| `optimalityTolerance` | First-order optimality tolerance | `1e-9` |

Use finite, real, double-precision data with aligned samples and a nonconstant phenotype. Handle missing values before fitting. The core functions assume valid inputs and do not perform comprehensive input validation. Constant features are returned unchanged; their correlation diagnostics remain undefined (`NaN`). COMET does not require positive input values mathematically, but the preprocessing and scale of the input must be specified for the intended analysis.

### Outputs

`matZ` has the same dimensions and sample/feature order as `matX`. The second output, called `model` in the examples and `stats` in the function signature, contains the fitted maps and diagnostics:

| Field | Meaning |
| --- | --- |
| `arrKnotX{j}` | Sorted distinct input values for feature j |
| `arrKnotZ{j}` | Corresponding fitted transformed values |
| `rvRawPCC` | Pearson correlation of each input feature with the phenotype |
| `rvPCC` | Pearson correlation after transformation |
| `rvSpearman` | Spearman correlation between input and transformed features |
| `rvDeltaNorm` | Euclidean norm of the change in feature values |
| `rvDirection` | Selected association direction, +1 or -1, for nonconstant features |
| `rvLoss` | Selected penalized objective value for nonconstant features |

The original implementation additionally returns `rvRelativeNorm`, the normalized squared displacement. The stable implementation instead adds `rvNumericalFloorHits`, `rvUniqueInput`, and `rvUniqueOutput`. To calculate normalized displacement for either implementation:

```matlab
P = sum((Ztrain - Xtrain).^2, 1) ./ ...
    sum((Xtrain - mean(Xtrain, 1)).^2, 1);
% P is undefined for constant input features.
```

## Applying fitted transformations to new samples

Keep the training feature order and preprocessing unchanged. Use linear interpolation between fitted nodes and endpoint linear extrapolation outside the training range:

```matlab
Znew = zeros(size(Xnew));
for j = 1:size(Xnew, 2)
    xk = model.arrKnotX{j};
    zk = model.arrKnotZ{j};
    if numel(xk) == 1
        Znew(:, j) = Xnew(:, j); % Identity for a constant training feature.
    else
        Znew(:, j) = interp1(xk, zk, Xnew(:, j), 'linear', 'extrap');
    end
end
```

No new-sample phenotype is used in this step. Record the fraction of new observations outside the training range: extrapolated values are less directly supported by the training data. The source functions fit the transformation; application to new data is illustrated above and in `example_COMET.m`.

## Algorithm

For each feature x, COMET:

1. Sorts its distinct input abundances into nodes c1 < ... < cm.
2. Represents transformed nodes through cumulative positive increments: q1 = theta1 and qk = q(k-1) + softplus(thetak) + epsilon, where epsilon is a positive minimum gap.
3. Optimizes correlation with the original phenotype while penalizing normalized squared displacement.
4. Returns the transformed observations and node pairs for subsequent application.

The objective is to minimize

```text
J_s(theta) = -s * corr(z(theta), y)
             + lambda * sum((z(theta) - x).^2) / sum((x - mean(x)).^2),

s in {-1, +1}.
```

The two directions are optimized separately from the identity transformation using quasi-Newton optimization; the solution with the smaller objective is retained. The map itself remains increasing for either direction. A negative feature–phenotype association does not require reversing abundance ranks. This is numerical optimization, not a guarantee of a global optimum.

## Interpretation and use

COMET is intended for monotonic feature–phenotype relationships whose curvature limits their representation by linear correlation. It is not a general correction for nonmonotonic relationships, missing values, or technical variation.

Because COMET uses the phenotype during fitting, an increase in training-set correlation is descriptive and does not by itself establish a biological association or improved prediction. For held-out evaluation, fit transformations and select features using training samples only. Within cross-validation, repeat those steps within each training fold before applying the fitted maps to the held-out fold.

Strict rank preservation does not preserve concentration differences, fold changes, or correlations between metabolites. Interpret those quantities on the appropriate original measurement scale. The penalty constrains changes in values but does not guarantee improvement on every downstream analysis.

## Manuscript and contact

This code accompanies the COMET manuscript. A journal citation and DOI should be added when available; no publication status is asserted here.

Correspondence: Jiyang Dong, Xiamen University — jydong@xmu.edu.cn.

## License

A license has not yet been specified for this deposit. The authors should select and add a `LICENSE` file before public release; this README does not grant a software license.
