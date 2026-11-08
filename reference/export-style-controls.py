#!/usr/bin/env python3
"""Matched CPU eager states and actual upstream default-SDPA style controls."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import subprocess
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from tokenizers import Tokenizer
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True)
p.add_argument('--input',type=Path,default=Path('refdata/conditioning'));p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.upstream/'src'))
assert subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()=='e27fa7fa0da7c9817879cc6210cc5693f9327946'
from mulacover import MuLaCoverGenPipeline,MuLaCoverGenConfig,MuLaCoverConfig
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
s=json.loads(Path('models/style.json').read_text());g=json.loads(Path('models/generator.json').read_text());root=Path(s['snapshot']);gen=Path(g['snapshot'])
model=AutoModel.from_pretrained(root,dtype=torch.float32,local_files_only=True,attn_implementation='eager').eval().to(a.device)
tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,padding_side='left');devices=dict.fromkeys(('mulacover','qwen','codec','transcriptor'),torch.device(a.device))
pipe=MuLaCoverGenPipeline({'mulacover':gen,'qwen':root},devices,dict.fromkeys(devices,torch.float32),Tokenizer.from_file(str(gen/'tokenizer.json')),MuLaCoverGenConfig.from_file(gen/'gen_config.json'),MuLaCoverConfig.from_pretrained(gen),lazy_load=True)
pipe._qwen=model;pipe._qwen_tokenizer=tokenizer;pipe.lazy_load=False
manifest=json.loads((a.input/'manifest.json').read_text());manifest['device']=a.device;manifest['stock_control']='actual upstream pipeline._encode_style with default SDPA';a.output.mkdir(parents=True,exist_ok=True)
for case in manifest['cases']:
    dest=a.output/case['name'];dest.mkdir(exist_ok=True)
    for name in ['lyrics.txt','tags.txt','normalized-lyrics.txt','normalized-tags.txt','tokens.i32','mask.u8','style-ids.i32','muq.f32']:shutil.copyfile(a.input/case['name']/name,dest/name)
    tag=(dest/'normalized-tags.txt').read_text()
    def save(name,tensor):np.ascontiguousarray(tensor.detach().cpu().numpy()).tofile(dest/name)
    with torch.inference_mode():
        if tag:
            model.set_attn_implementation('eager');batch=tokenizer([tag],return_tensors='pt',padding=True,truncation=False);out=model(**{k:v.to(a.device) for k,v in batch.items()},output_hidden_states=True)
            save('qwen.f32',torch.nn.functional.normalize(out.last_hidden_state[:,-1].float(),dim=-1)[0]);save('style-hidden.f32',out.last_hidden_state)
            for index in [0,1,14,28]:save(f'style-layer-{index}.f32',out.hidden_states[index])
        else:save('qwen.f32',torch.zeros(1024))
        model.set_attn_implementation('sdpa');save('qwen-stock.f32',pipe._encode_style(tag))
    print('Frozen',a.device,case['name'],flush=True)
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
