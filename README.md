# COMET and phenotype-guided linearization

Algorithm code for **Data Transformations for Enhancing the Linear Visibility of Metabolite–Phenotype Associations in Metabolomics**.

This package provides ACE-M, MOS and COMET for learning metabolite abundance transformations from a continuous phenotype. ACE-M uses a monotone smoother with the phenotype scale fixed. MOS uses isotonic scores and can introduce plateaus. COMET enforces strictly increasing fitted abundance levels and penalizes departure from the input.

Conventional transformations such as log1p, square root and Box–Cox address distributional properties. Phenotype-guided linearization addresses the expression of metabolite–phenotype associations in linear analysis. It can be applied to the original representation or after a conventional transformation; any additional benefit requires validation on the intended task.

## Contents

```text
Algorithm/                         Python, R and MATLAB functions
Examples/example_family.py         One runnable example with two data options
Examples/fit_ace_example.R          R helper for the same example
Tests/                             Algorithm and test-data checks
DataSets/CRC_data.xlsx             CRC workbook
DataSets/Prepared/Pregnancy/        Prepared pregnancy test data and identifiers
Docs/USAGE_zh.md                   Detailed usage guide in Chinese
Docs/Third_party/                  Retained third-party notices
```

The package contains no precomputed analysis results, manuscript figures or full-paper experiment pipelines. Example execution prints checks to the console and does not save analysis outputs.

## Install

The Python/R example uses Python 3.12 and R with base `stats`. No `acepack` installation or MATLAB installation is needed for this example.

```bash
python -m pip install -r requirements.txt
```

Put `Rscript` on `PATH`, or provide its full path with `--rscript`. Run the commands below from the repository root.

## Run the example

```bash
python Examples/example_family.py --dataset pregnancy --rscript Rscript
```

This fits the first two prepared pregnancy features using 507 profiles from 21 training participants, then applies frozen maps to 245 profiles from nine other participants. It checks output dimensions, fitted order, finite values and endpoint clamping. Validation gestational ages are not passed to a map.

The same example can read the CRC workbook:

```bash
python Examples/example_family.py --dataset crc --rscript Rscript
```

This uses two log1p-transformed metabolites from 92 healthy controls, a simulated continuous response and a fixed 64/28 sample split. The response is generated for demonstration; disease labels are not used as the response. Neither data option is a reproduction of the manuscript's performance evaluation.

On Windows, for example:

```powershell
python Examples/example_family.py --dataset pregnancy --rscript "C:/Program Files/R/R-4.6.1/bin/Rscript.exe"
```

## Fit and apply a map

```python
import numpy as np
from Algorithm.comet_numpy import fit_projected
from Algorithm.apply_map import apply_map

x_train = np.linspace(0.1, 2.0, 40)
y_train = np.sqrt(x_train)
z_train, diagnostics = fit_projected(x_train, y_train, lam=0.5)
assert all(fit['success'] for fit in diagnostics)
z_new = apply_map(x_train, z_train, np.array([0.05, 0.5, 2.5]))
```

For MOS, replace the fitting call with `mos(x_train, y_train)` from `Algorithm.mos_numpy`. For ACE-M, use `ace_fixed_y_classic(x_train, y_train)$tx` after sourcing the supplied R file; the example shows the Python/R interchange. See [the interface list](Algorithm/README.md) and [the usage guide](Docs/USAGE_zh.md) for MATLAB calls and data loading.

## Run tests

```bash
python -m unittest discover -s Tests -p "test_*.py" -v
Rscript --vanilla Tests/test_ace.R
```

Tests cover the COMET objective and gradient, order, equal-input handling, MOS plateaus, interpolation, endpoint clamping, ACE-M conventions, and alignment of the two distributed datasets. Test execution does not write result files.

## Input and application rules

- Matrices have samples in rows and metabolites in columns. Scalar fitting kernels take finite, aligned one-dimensional NumPy arrays with nonconstant feature and phenotype values.
- Fit transformations, missing-value handling, filtering, scaling and model selection using training data only. Apply the saved rules to new samples without using their phenotypes.
- Maps interpolate linearly within training support and clamp to the fitted endpoint outside it. Clamping can introduce boundary ties. COMET's strict increase applies within training support.
- Transformed values need not be nonnegative. Order preservation does not preserve concentration differences, fold changes or between-metabolite correlations.
- Inspect convergence diagnostics. Training correlation alone does not establish downstream analytical benefit.

## Data, attribution and license status

Only the CRC workbook and prepared pregnancy dataset are distributed. [DataSets/README.md](DataSets/README.md) describes their sources, orientation and scope. The pregnancy panel is training-selected and does not represent the entire metabolome.

No new license has been assigned to the authors' original code. Third-party permissions and dataset terms remain applicable; see [LICENSE_NOTICE.md](LICENSE_NOTICE.md). Software citation metadata are supplied in `CITATION.cff`; add the publication DOI when available.

Repository: [BioNet-XMU/COMET](https://github.com/BioNet-XMU/COMET). Correspondence: Jiyang Dong, Xiamen University, jydong@xmu.edu.cn.
