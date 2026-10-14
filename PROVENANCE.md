# Source provenance

The algorithm files were copied without modification from the manuscript workspace. Only the README and usage example were newly written for this deposit.

## transform_COMET_continuous.m
- Workspace source: `Code/transform_COMET_continuous.m`
- SHA-256: `720b0a00a137d5ac8d347e50e4d23ae7162d70dd0c82747537684211b502f0a7`

## transform_COMET_continuous_stable.m
- Workspace source: `Code/transform_COMET_continuous_stable.m`
- SHA-256: `d4174c3b69fbb91c67cd40dcff138da616595fc3faf735a31bc36c72d7135dc5`

The original implementation was used in the manuscript simulations. The stable implementation was used in the independent-participant pregnancy analysis and its penalty-sensitivity analysis. Both are provided to preserve this distinction.

The original Chinese help comments mention default parameter values, but omitted arguments are not implemented. Follow the explicit seven-argument calls in README.md.

Validation: see the MATLAB example, which checks finite outputs, strictly increasing fitted knots, retained input ties, constant-feature identity, and reapplication of the training map.
