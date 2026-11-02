"""Verify actual GGUF block storage, preserved auxiliary tensors and refusal paths."""
import json
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import gguf

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'build-windows-development-cpu/bin/Release/mulacover-quantize.exe'
CLI=Path(os.environ.get('MULACOVER_QUANTIZE',str(DEFAULT)))


@unittest.skipUnless(CLI.is_file(),'Build mulacover-quantize or set MULACOVER_QUANTIZE')
class Quantization(unittest.TestCase):
    def source(self,path):
        writer=gguf.GGUFWriter(str(path),'mulacover')
        writer.add_uint32('mulacover.format_version',1)
        writer.add_string('mulacover.weight_precision','F32')
        shapes={'backbone.layers.0.attn.q_proj.weight':(17,512),
                'backbone.layers.0.attn.v_proj.weight':(17,512),
                'backbone.layers.0.mlp.w2.weight':(33,512),
                'audio_embeddings.weight':(33,512),'codebook0_head.weight':(33,512),
                'small.weight':(17,64),'single.weight':(1,512),
                'backbone.norm.scale':(512,),'rope.cache.64':(33,64),
                'peft_layers.20.adaptor.chord_proj.weight':(24,12)}
        values={name:np.sin(np.arange(np.prod(shape),dtype=np.float32)*0.03).reshape(shape) for name,shape in shapes.items()}
        for name,value in values.items():writer.add_tensor(name,value)
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
        return values

    def run_cli(self,source,out,mix='q4_k_m'):
        return subprocess.run([str(CLI.resolve()),'--input',str(source),'--output',str(out),'--mix',mix],capture_output=True,text=True)

    def test_block_storage_and_preserved_auxiliary_values(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as cleanup:
            source,out=Path(temp)/'f32.gguf',Path(temp)/'q4.gguf'
            values=self.source(source);run=self.run_cli(source,out)
            self.assertEqual(run.returncode,0,run.stderr)
            reader=gguf.GGUFReader(str(out));tensors={t.name:t for t in reader.tensors}
            cleanup.callback(reader.data._mmap.close)
            for name,t in tensors.items():
                kind=t.tensor_type
                if name=='backbone.layers.0.attn.q_proj.weight':self.assertEqual(kind,gguf.GGMLQuantizationType.Q4_K)
                elif name in ('backbone.layers.0.attn.v_proj.weight','backbone.layers.0.mlp.w2.weight','audio_embeddings.weight','codebook0_head.weight'):
                    self.assertEqual(kind,gguf.GGMLQuantizationType.Q6_K)
                else:
                    self.assertEqual(kind,gguf.GGMLQuantizationType.F32)
                    np.testing.assert_array_equal(np.asarray(t.data).reshape(-1),values[name].reshape(-1))
                block,size=gguf.GGML_QUANT_SIZES[kind]
                self.assertEqual(t.n_bytes,int(np.prod(t.shape))*size//block)
                self.assertEqual(t.data_offset%32,0)
            report=json.loads(Path(str(out)+'.quantization.json').read_text())
            self.assertEqual(report['file_bytes'],out.stat().st_size)
            self.assertEqual(report['tensor_count'],len(values))
            self.assertEqual(report['policy'],'mulacover-kquant-v1')
            self.assertLess(out.stat().st_size,source.stat().st_size)
            del tensors,reader

    def test_refuses_requantization_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source,out=Path(temp)/'f32.gguf',Path(temp)/'q4.gguf';self.source(source)
            original=source.read_bytes();self.assertNotEqual(self.run_cli(source,source).returncode,0)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(self.run_cli(source,out).returncode,0)
            quantized=out.read_bytes();self.assertNotEqual(self.run_cli(source,out).returncode,0)
            self.assertEqual(out.read_bytes(),quantized)
            self.assertNotEqual(self.run_cli(out,Path(temp)/'twice.gguf').returncode,0)
            self.assertFalse((Path(temp)/'twice.gguf').exists())


if __name__=='__main__':unittest.main()
