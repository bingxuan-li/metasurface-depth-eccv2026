import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
from metasurface_depth.infer import preprocess


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def pair(self, a, b):
        paths = self.root / "a.png", self.root / "b.png"
        for path, arr in zip(paths, (a, b)):
            self.assertTrue(cv2.imwrite(str(path), arr))
        return paths

    def test_channels_and_center_crop(self):
        a = np.arange(31 * 43, dtype=np.uint16).reshape(31, 43).astype(np.uint8)
        b = np.flip(a, axis=1).copy()
        tensor, meta = preprocess(*self.pair(a, b))
        ta = torch.from_numpy(a.astype(np.float32)[1:29, :42] / 255)
        tb = torch.from_numpy(b.astype(np.float32)[1:29, :42] / 255)
        self.assertTrue(torch.equal(tensor, torch.stack((ta, tb, (ta + tb) / 2)).unsqueeze(0)))
        self.assertEqual(meta["crop_xywh"], [0, 1, 42, 28])

    def test_exact_multiple(self):
        tensor, meta = preprocess(
            *self.pair(np.zeros((28, 42), np.uint8), np.full((28, 42), 255, np.uint8))
        )
        self.assertEqual(meta["crop_xywh"], [0, 0, 42, 28])
        self.assertTrue(torch.all(tensor[:, 2] == 0.5))

    def test_reject_16bit(self):
        with self.assertRaisesRegex(ValueError, "uint8"):
            preprocess(*self.pair(np.zeros((28, 28), np.uint16), np.zeros((28, 28), np.uint16)))

    def test_reject_mismatch(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            preprocess(*self.pair(np.zeros((28, 28), np.uint8), np.zeros((28, 42), np.uint8)))

    def test_reject_tiny(self):
        with self.assertRaisesRegex(ValueError, "14x14"):
            preprocess(*self.pair(np.zeros((13, 28), np.uint8), np.zeros((13, 28), np.uint8)))

    def test_reject_missing(self):
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            preprocess(self.root / "missing.png", self.root / "also_missing.png")


if __name__ == "__main__":
    unittest.main()
