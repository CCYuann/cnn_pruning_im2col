import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_adapted_gradient_study as study


class AdaptedGradientReportTests(unittest.TestCase):
    def test_control_seed_is_not_passed_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for c in ["l1_sum", "top2_mean_abs"]:
                for g in ["flat", "spatial"]:
                    directory = root/f"resnet50_seed0_{c}_{g}/pilots"
                    directory.mkdir(parents=True)
                    (directory/(c+".json")).write_text(json.dumps(dict(seed=0, variant=c, final_validation_acc=.93)))
            with patch.object(study, "BASE", root):
                rows = study.baseline_rows(0)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["seed"] == 0 for r in rows))

    def test_promotion_requires_best_control_and_both_ablations(self):
        baseline = [dict(final_validation_acc=.9368)]
        values = dict(gradient_gate=.9378, gradient_all=.9377, matched_random=.9377)
        with patch.object(study, "baseline_rows", return_value=baseline), patch.object(study, "metric", side_effect=lambda seed, variant: dict(final_validation_acc=values[variant])):
            self.assertTrue(study.promising())
            values["matched_random"] = .938
            self.assertFalse(study.promising())


if __name__ == "__main__":
    unittest.main()
