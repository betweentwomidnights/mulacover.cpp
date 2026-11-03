import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from continuation import frames_for_seconds,select_guide,plan_frames,write_continuation


class ContinuationTest(unittest.TestCase):
    def test_selection_uses_requested_source_region_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'codes.i32';codes=np.arange(128*8,dtype='<i4').reshape(-1,8);codes.tofile(p);original=p.read_bytes()
            selected,meta=select_guide(p,'4','2')
            np.testing.assert_array_equal(selected,codes[25:75]);self.assertEqual(meta['guide_frames'],50)
            self.assertEqual(meta['source_start_seconds'],2);self.assertEqual(p.read_bytes(),original)
            self.assertEqual(meta['source_codes_sha256'],hashlib.sha256(original).hexdigest())
            with self.assertRaises(ValueError):select_guide(p,'16','0')
            p.write_bytes(original[:-1])
            with self.assertRaises(ValueError):select_guide(p)

    def test_exact_time_grid_and_cache(self):
        self.assertEqual(frames_for_seconds('7.92'),99)
        self.assertEqual(frames_for_seconds('7.99'),99)
        self.assertEqual(plan_frames(25,99,continuation_seconds='2.32',capacity=1024),(128,1024))
        self.assertEqual(plan_frames(25,100,continuation_seconds='30'),(475,512))
        self.assertEqual(plan_frames(25,0),(100,256))
        for duration in ('NaN','Infinity','-1','0','.079'):
            with self.assertRaises(ValueError):frames_for_seconds(duration)
        with self.assertRaises(ValueError):plan_frames(25,100,frames=475,continuation_seconds='30')
        with self.assertRaises(ValueError):plan_frames(25,100,continuation_seconds='30',capacity=256)
        with self.assertRaises(ValueError):plan_frames(25,8190,continuation_seconds='1')

    def test_suffix_export_preserves_samples_without_guide(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'full.wav';out=Path(d)/'continuation.wav'
            samples=np.arange(3*3840*2,dtype='<f4');data=samples.tobytes()
            header=struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',36+len(data),b'WAVE',b'fmt ',16,3,2,48000,384000,8,32,b'data',len(data))
            p.write_bytes(header+data);write_continuation(p,out,1,3)
            self.assertEqual(out.read_bytes()[44:],data[3840*8:])
            self.assertEqual(struct.unpack_from('<I',out.read_bytes(),40)[0],2*3840*8)
            self.assertEqual(p.read_bytes(),header+data)
            with self.assertRaises(ValueError):write_continuation(p,p,1,3)


if __name__=='__main__':unittest.main()
