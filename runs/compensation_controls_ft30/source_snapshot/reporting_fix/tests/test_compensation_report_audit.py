import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from finalize_compensation_controls import equal_nested


class ReportAuditTest(unittest.TestCase):
    def test_mixed_rng_state(self):
        a=dict(python=(3,(1,2),None),numpy=("MT19937",np.array([1,2,3],dtype=np.uint32),0,0,0.),
               torch=torch.tensor([1,2],dtype=torch.uint8))
        b=dict(python=(3,(1,2),None),numpy=("MT19937",np.array([1,2,3],dtype=np.uint32),0,0,0.),
               torch=torch.tensor([1,2],dtype=torch.uint8))
        self.assertTrue(equal_nested(a,b))
        b["numpy"][1][1]=4
        self.assertFalse(equal_nested(a,b))

    def test_type_shape_and_key_mismatch(self):
        self.assertFalse(equal_nested(np.ones(2),np.ones(3)))
        self.assertFalse(equal_nested(torch.ones(2),torch.ones(3)))
        self.assertFalse(equal_nested({"a":1},{"b":1}))


if __name__=="__main__":
    unittest.main()
