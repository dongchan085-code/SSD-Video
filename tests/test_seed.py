"""Quick determinism checks for ssd_vlm.utils.seed."""

import random
import unittest

import numpy as np
import torch

from ssd_vlm.utils.seed import seed_worker, set_global_seed


class TestSetGlobalSeed(unittest.TestCase):
    def test_returns_masked_seed(self):
        applied = set_global_seed(42)
        self.assertEqual(applied, 42)

    def test_seeds_python_numpy_torch(self):
        set_global_seed(123)
        py_first = random.random()
        np_first = float(np.random.rand())
        torch_first = torch.rand(1).item()

        set_global_seed(123)
        self.assertAlmostEqual(random.random(), py_first)
        self.assertAlmostEqual(float(np.random.rand()), np_first)
        self.assertAlmostEqual(torch.rand(1).item(), torch_first)

    def test_seed_worker_does_not_raise(self):
        # Driver thread has a torch initial_seed; worker_init_fn relies on it.
        torch.manual_seed(7)
        seed_worker(0)
        seed_worker(3)


if __name__ == "__main__":
    unittest.main()
