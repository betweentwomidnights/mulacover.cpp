import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('tempo_analysis',Path(__file__).resolve().parents[1]/'tools/tempo-analysis.py');tempo=importlib.util.module_from_spec(spec);spec.loader.exec_module(tempo)


class TempoCalibration(unittest.TestCase):
    def test_known_click_pulses_include_correct_period_among_aliases(self):
        for bpm in (80,120,160):
            with self.subTest(bpm=bpm):
                fps=100;env=np.zeros(3000)
                for t in np.arange(.3,29,60/bpm):env[round(t*fps)]=1
                candidates=tempo.tempo_candidates(env,fps)
                self.assertTrue(any(abs(c['bpm']-bpm)/bpm<.025 for c in candidates),candidates)

    def test_silence_does_not_claim_a_tempo(self):
        self.assertEqual(tempo.tempo_candidates(np.zeros(3000),100),[])


if __name__=='__main__':unittest.main()
