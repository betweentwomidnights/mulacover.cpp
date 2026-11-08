import importlib.util
from pathlib import Path
import sys
import unittest
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools'))
spec=importlib.util.spec_from_file_location('score_alignment',root/'tools/score-alignment.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ScoreAlignmentTest(unittest.TestCase):
    def test_octaves_allowed_but_wrong_pitch_or_late_onset_rejected(self):
        reference=[(0,60),(1,64),(2,67)]
        self.assertEqual(module.match(reference,[(.02,72),(1.1,76),(2.5,79)],.25),[(0,0),(1,1)])
        self.assertEqual(module.match(reference,[(.02,61),(1.1,65),(2,68)],.25),[])

    def test_one_to_one_and_order(self):
        self.assertEqual(len(module.match([(0,60)],[(0,60),(.1,72)],.25)),1)
        self.assertEqual(len(module.match([(0,60),(0,64)],[(0,64),(0,60)],.25)),1)
        self.assertEqual(module.match([],[(0,60)],.25),[])


if __name__=='__main__':unittest.main()
