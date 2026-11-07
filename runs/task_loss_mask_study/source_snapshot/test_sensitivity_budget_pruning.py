import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sensitivity_budget_pruning import allocate_counts


def cluster(k=100, reference=.5, sensitivity=0, limit=99, filters=8):
    return {"positions": k, "reference_ratio": reference, "sensitivity": sensitivity,
            "max_prune": limit, "filters": filters}


class AllocationTests(unittest.TestCase):
    def test_protection_and_exact_budget(self):
        groups = [cluster(limit=0), cluster(limit=20), cluster(), cluster()]
        counts = allocate_counts(groups, 200 * 8)
        self.assertEqual(sum(counts) * 8, 1600)
        self.assertEqual(counts[0], 0)
        self.assertLessEqual(counts[1], 20)
        self.assertTrue(all(c < 100 for c in counts))

    def test_less_sensitive_receives_more_extra_pruning(self):
        counts = allocate_counts([cluster(sensitivity=0), cluster(sensitivity=9)], 140 * 8)
        self.assertGreater(counts[0], counts[1])
        self.assertEqual(sum(counts), 140)

    def test_unrepresentable_or_infeasible_budget(self):
        with self.assertRaisesRegex(ValueError, "not representable"):
            allocate_counts([cluster()], 401)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            allocate_counts([cluster(limit=20)], 21 * 8)
        with self.assertRaisesRegex(ValueError, "partial"):
            allocate_counts([cluster(), cluster(filters=4)], 100 * 8)

    def test_seeded_varied_k_sizes_and_caps(self):
        rng = random.Random(12)
        for _ in range(100):
            groups = []
            for _ in range(20):
                k = rng.choice([9, 27, 64, 144, 512])
                groups.append(cluster(k=k, reference=rng.random(), sensitivity=rng.random()*50,
                                      limit=rng.choice([0, int(.2*k), k-1])))
            initial = [min(round(c["reference_ratio"] * c["positions"]), c["max_prune"]) for c in groups]
            target = rng.randint(sum(initial), sum(c["max_prune"] for c in groups))
            counts = allocate_counts(groups, target * 8)
            self.assertEqual(sum(counts), target)
            self.assertTrue(all(start <= count <= c["max_prune"] for start, count, c in zip(initial, counts, groups)))
            self.assertEqual(counts, allocate_counts(groups, target * 8))


if __name__ == "__main__":
    unittest.main()
