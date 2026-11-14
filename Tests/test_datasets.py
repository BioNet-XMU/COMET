"""Check that the distributed test inputs retain their alignment and split."""
from pathlib import Path
import csv
import unittest
import numpy as np
from scipy.io import loadmat
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]


class TestDatasets(unittest.TestCase):
    def test_crc_layout(self):
        workbook = load_workbook(ROOT / 'DataSets/CRC_data.xlsx', read_only=True, data_only=True)
        rows = list(workbook.active.values)
        workbook.close()
        self.assertEqual((len(rows) - 1, len(rows[0]) - 2), (113, 234))
        self.assertEqual(sum(str(s).startswith('H') for s in rows[0][2:]), 92)
        self.assertTrue(np.isfinite(np.asarray([r[2:] for r in rows[1:]], dtype=float)).all())

    def test_pregnancy_alignment_and_independent_participants(self):
        folder = ROOT / 'DataSets/Prepared/Pregnancy'
        data = loadmat(folder / 'data.mat', simplify_cells=True)
        self.assertEqual({k for k in data if not k.startswith('__')}, {
            'Xtrain', 'Xvalidation', 'ytrain', 'yvalidation',
            'subject_train', 'subject_validation', 'sample_train', 'sample_validation'})
        self.assertEqual(data['Xtrain'].shape, (507, 171))
        self.assertEqual(data['Xvalidation'].shape, (245, 171))
        with (folder / 'features.csv').open(encoding='utf-8', newline='') as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 171)
        with (folder / 'samples.csv').open(encoding='utf-8', newline='') as handle:
            samples = list(csv.DictReader(handle))
        self.assertEqual(len(samples), 752)
        for split in ['train', 'validation']:
            rows = [row for row in samples if row['split'] == split]
            np.testing.assert_array_equal([int(r['sample_id']) for r in rows], data['sample_' + split])
            np.testing.assert_array_equal([r['subject'] for r in rows], data['subject_' + split])
            np.testing.assert_allclose([float(r['ga']) for r in rows], data['y' + split])
            self.assertTrue(np.isfinite(data['X' + split]).all())
        self.assertEqual(len(set(data['subject_train'])), 21)
        self.assertEqual(len(set(data['subject_validation'])), 9)
        self.assertTrue(set(data['subject_train']).isdisjoint(data['subject_validation']))


if __name__ == '__main__':
    unittest.main()
