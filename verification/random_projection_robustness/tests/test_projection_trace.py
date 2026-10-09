import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(ROOT))
from src.analysis.frozen_mechanisms import rademacher_jacobian_energy


class ProjectionTraceTests(unittest.TestCase):
    def test_trace_records_exact_signs_without_changing_energy(self):
        raw = torch.tensor([[[0.2], [0.4]], [[-0.3], [0.5]]], dtype=torch.float32)
        forward_a = lambda x: x.square()
        forward_b = lambda x: 2.0 * x
        seed = 17
        count = 3

        expected_generator = torch.Generator(device="cpu").manual_seed(seed)
        expected = [
            torch.randint(0, 2, (2, 2), generator=expected_generator).mul_(2).sub_(1)
            for _ in range(count)
        ]

        captured = []
        traced_generator = torch.Generator(device="cpu").manual_seed(seed)
        traced = rademacher_jacobian_energy(
            forward_a,
            forward_b,
            raw,
            projections=count,
            generator=traced_generator,
            projection_recorder=lambda index, signs: captured.append((index, signs.clone())),
        )

        plain_generator = torch.Generator(device="cpu").manual_seed(seed)
        plain = rademacher_jacobian_energy(
            forward_a, forward_b, raw, projections=count, generator=plain_generator
        )

        self.assertEqual([index for index, _ in captured], list(range(count)))
        for (_, actual), expected_signs in zip(captured, expected):
            self.assertTrue(torch.equal(actual, expected_signs))
        for actual, baseline in zip(traced, plain):
            self.assertTrue(torch.equal(actual, baseline))


if __name__ == "__main__":
    unittest.main()
