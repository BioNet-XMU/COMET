"""Fit two features from either test dataset and apply frozen maps."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import tempfile
import numpy as np
from scipy.io import loadmat
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Algorithm.apply_map import apply_map
from Algorithm.comet_numpy import fit_projected
from Algorithm.mos_numpy import mos


def load_data(dataset):
    if dataset == 'pregnancy':
        data = loadmat(ROOT / 'DataSets/Prepared/Pregnancy/data.mat', simplify_cells=True)
        return data['Xtrain'][:, :2], data['ytrain'], data['Xvalidation'][:, :2]
    workbook = load_workbook(ROOT / 'DataSets/CRC_data.xlsx', read_only=True, data_only=True)
    rows = list(workbook.active.values)
    workbook.close()
    controls = [i for i, name in enumerate(rows[0][2:]) if str(name).startswith('H')]
    raw = np.asarray([row[2:] for row in rows[1:]], dtype=float).T[controls]
    rng = np.random.default_rng(20261114)
    # A simulated response, not the CRC clinical label or an observed phenotype.
    y = np.sqrt(raw[:, 0] / np.median(raw[:, 0])) + .05 * rng.normal(size=len(raw))
    order = rng.permutation(len(raw))
    train, test = order[:64], order[64:]
    X = np.log1p(raw[:, :2])
    return X[train], y[train], X[test]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['pregnancy', 'crc'], default='pregnancy')
    parser.add_argument('--rscript', default='Rscript', help='Rscript executable or full path')
    args = parser.parse_args()
    Xtrain, ytrain, Xnew = load_data(args.dataset)
    assert np.isfinite(Xtrain).all() and np.isfinite(ytrain).all()
    assert np.isfinite(Xnew).all() and np.all(np.std(Xtrain, axis=0) > 0)
    env = os.environ.copy()
    if os.name == 'nt':
        env['LC_ALL'] = 'English_United States.utf8'
    # R interchange files are temporary and are removed when this block exits.
    with tempfile.TemporaryDirectory(prefix='comet_example_') as directory:
        temporary = Path(directory)
        np.savetxt(temporary / 'X.csv', Xtrain, delimiter=',')
        np.savetxt(temporary / 'y.csv', ytrain, delimiter=',')
        subprocess.run([args.rscript, '--vanilla', str(ROOT / 'Examples/fit_ace_example.R'),
                        str(ROOT), str(temporary / 'X.csv'), str(temporary / 'y.csv'),
                        str(temporary / 'ACE.csv')], cwd=ROOT, env=env, check=True)
        ace = np.loadtxt(temporary / 'ACE.csv', delimiter=',', ndmin=2)
    fitted = {'ACE-M': ace, 'MOS': Xtrain.copy(), 'COMET': Xtrain.copy()}
    for j in range(Xtrain.shape[1]):
        fitted['MOS'][:, j] = mos(Xtrain[:, j], ytrain)
        fitted['COMET'][:, j], diagnostics = fit_projected(Xtrain[:, j], ytrain, lam=.5)
        if not all(fit['success'] for fit in diagnostics):
            raise RuntimeError('COMET did not meet its stopping tolerance; inspect the diagnostics.')
    for method, training_scores in fitted.items():
        transformed = np.column_stack([
            apply_map(Xtrain[:, j], training_scores[:, j], Xnew[:, j])
            for j in range(Xtrain.shape[1])])
        assert transformed.shape == Xnew.shape and np.isfinite(transformed).all()
        for j in range(Xtrain.shape[1]):
            _, first = np.unique(Xtrain[:, j], return_index=True)
            gaps = np.diff(training_scores[first, j])
            if method == 'COMET':
                assert np.all(gaps > 0)
            else:
                assert np.all(gaps >= -1e-12 * max(1., np.ptp(training_scores[:, j])))
            endpoints = apply_map(Xtrain[:, j], training_scores[:, j], [-np.inf, np.inf])
            np.testing.assert_array_equal(endpoints, training_scores[first[[0, -1]], j])
        print(f'{method}: fitted {Xtrain.shape}; applied to {transformed.shape}; boundary checks passed.')
    print(f'{args.dataset}: complete. No new-sample phenotype was supplied to a map.')
    print('No result files were saved. This is a usage example, not a performance benchmark.')


if __name__ == '__main__':
    main()
