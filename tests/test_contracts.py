import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from metasurface_depth.data import EncodedDataset, assert_disjoint
from metasurface_depth.demo import create_demo
from metasurface_depth.infer import preprocess
from metasurface_depth.io import read_manifest, sha256
from metasurface_depth.metrics import gradient_loss, metrics, training_loss
from metasurface_depth.simulator.api import PSF_SHA256


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, name="sample"):
        for i in (1, 2):
            cv2.imwrite(str(self.root / f"{name}{i}.png"), np.full((31, 43), 30 * i, np.uint8))
        np.save(self.root / f"{name}.npy", np.full((31, 43), 0.5, np.float32))
        manifest = self.root / f"{name}.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image1", "image2", "depth"])
            writer.writerow([f"{name}1.png", f"{name}2.png", f"{name}.npy"])
        return manifest

    def test_data_inference_parity(self):
        path = self.dataset()
        image, depth = EncodedDataset(path)[0]
        expected, _ = preprocess(self.root / "sample1.png", self.root / "sample2.png")
        self.assertTrue(torch.equal(image, expected[0]))
        self.assertEqual(depth.shape, (1, 28, 42))

    def test_split_leakage(self):
        dataset = EncodedDataset(self.dataset())
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint([dataset], [dataset])
        assert_disjoint([dataset], [EncodedDataset(self.dataset("other"))])

    def test_metrics_and_loss(self):
        gt = torch.full((1, 1, 28, 28), 0.5)
        self.assertEqual(metrics(gt, gt)["delta_1"], 1)
        self.assertEqual(metrics(gt, gt)["mae"], 0)
        self.assertEqual(gradient_loss(gt, gt), 0)
        self.assertEqual(training_loss(gt, gt), 0)
        bad = gt.clone()
        bad[..., 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            training_loss(gt, bad)

    def test_gradient_matches_original_definition(self):
        torch.manual_seed(3)
        p, t = torch.rand(2, 1, 28, 42), torch.rand(2, 1, 28, 42)
        expected = (
            ((p[..., :-1] - p[..., 1:]).abs() - (t[..., :-1] - t[..., 1:]).abs()).abs().mean()
        )
        expected += (
            ((p[..., :-1, :] - p[..., 1:, :]).abs() - (t[..., :-1, :] - t[..., 1:, :]).abs())
            .abs()
            .mean()
        )
        self.assertTrue(torch.equal(expected, gradient_loss(p, t)))

    def test_demo_is_deterministic(self):
        a = create_demo(self.root / "a")
        b = create_demo(self.root / "b")
        for name in ("scene_00.png", "scene_00.npy"):
            self.assertEqual(sha256(a.parent / name), sha256(b.parent / name))
        self.assertEqual(len(read_manifest(a, ("rgb", "depth"))), 3)

    def test_bundled_psf(self):
        from importlib.resources import files

        path = Path(str(files("metasurface_depth") / "assets/psf_v5.pt"))
        self.assertEqual(sha256(path), PSF_SHA256)

    def test_empty_manifest_rejected(self):
        path = self.root / "empty.csv"
        path.write_text("rgb,depth\n")
        with self.assertRaises(ValueError):
            read_manifest(path, ("rgb", "depth"))


if __name__ == "__main__":
    unittest.main()
