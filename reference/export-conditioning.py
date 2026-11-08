#!/usr/bin/env python3
"""Offline pinned tokenizer assets, unquantized style weights and frontend goldens."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
import sys
import numpy as np
import torch
from gguf import GGUFWriter

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('refdata/conditioning'))
a=p.parse_args()
upstream_revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
assert upstream_revision=='e27fa7fa0da7c9817879cc6210cc5693f9327946'
sys.path.insert(0,str(a.upstream/'src'))
from mulacover import MuLaCoverGenPipeline,MuLaCoverGenConfig,MuLaCoverConfig
from tokenizers import Tokenizer
from transformers import AutoModel,AutoTokenizer
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
sources={n:json.loads(Path('models',n+'.json').read_text()) for n in ('generator','style')}
assert sources['generator']['revision']=='bbbaef2b31835c2ef5172ff528dbe325230d46fb'
assert sources['style']['revision']=='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
g=Path(sources['generator']['snapshot']);s=Path(sources['style']['snapshot'])
a.output.mkdir(parents=True,exist_ok=True)

assets=Path('models/mulacover-conditioning-tokenizers.gguf')
w=GGUFWriter(str(assets),'mulacover-tokenizers',use_temp_file=True)
w.add_uint32('mulacover.tokenizer_version',1)
for name,path,expected_vocab,digits,nfc,ignore in [('prompt',g,128000,3,False,True),('style',s,151643,1,True,False)]:
    data=json.loads((path/'tokenizer.json').read_text());model=data['model']
    assert model['type']=='BPE' and model.get('dropout') is None and not model.get('byte_fallback')
    assert bool(model.get('ignore_merges'))==ignore and len(model['vocab'])==expected_vocab
    assert (data['normalizer']=={'type':'NFC'})==nfc
    regex=data['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex']
    expected=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"+(r'\p{N}{1,3}' if digits==3 else r'\p{N}')+r'| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+'
    assert regex==expected
    vocab=['']*expected_vocab
    for token,index in model['vocab'].items():vocab[index]=token
    merges=[' '.join(x) if isinstance(x,list) else x for x in model['merges']]
    added=data['added_tokens'];assert all(not t['single_word'] and not t['lstrip'] and not t['rstrip'] and not t['normalized'] for t in added)
    for key,values in [('tokens',vocab),('merges',merges),('specials',[t['content'] for t in added])]:w.add_array(name+'.'+key,values)
    w.add_array(name+'.special_ids',[t['id'] for t in added]);w.add_uint32(name+'.digits',digits)
    w.add_bool(name+'.nfc',nfc);w.add_bool(name+'.ignore_merges',ignore)
    w.add_string(name+'.source_sha256',hashlib.sha256((path/'tokenizer.json').read_bytes()).hexdigest())
for k,v in vars(MuLaCoverGenConfig.from_file(g/'gen_config.json')).items():w.add_uint32('prompt.config.'+k,v)
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()

model=AutoModel.from_pretrained(s,dtype=torch.float32,local_files_only=True,attn_implementation='eager').eval().to('cuda')
config=model.config
assert (config.hidden_size,config.intermediate_size,config.num_attention_heads,config.num_key_value_heads,config.head_dim,config.num_hidden_layers,config.vocab_size)==(1024,3072,16,8,128,28,151669)
style_path=Path('models/mulacover-style-f32.gguf')
w=GGUFWriter(str(style_path),'mulacover-style',use_temp_file=True)
w.add_uint32('mulacover.style_version',1);w.add_string('mulacover.style_revision',sources['style']['revision'])
for key in ['hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','num_hidden_layers','vocab_size']:w.add_uint32('style.'+key,getattr(config,key))
w.add_float32('style.rms_norm_eps',config.rms_norm_eps);w.add_float32('style.rope_theta',config.rope_theta)
state=model.state_dict()
for key,value in sorted(state.items()):w.add_tensor(key,np.ascontiguousarray(value.detach().cpu().float().numpy()))
# Freeze exact HF rotary trig, avoiding runtime transcendental differences.
with torch.inference_mode():
    cos,sin=model.rotary_emb(torch.zeros(1,8192,1024,device='cuda'),torch.arange(8192,device='cuda').unsqueeze(0))
for key,value in [('rope.cos',cos[0]),('rope.sin',sin[0])]:w.add_tensor(key,np.ascontiguousarray(value.cpu().numpy()))
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()

text=Tokenizer.from_file(str(g/'tokenizer.json'));qtok=AutoTokenizer.from_pretrained(s,local_files_only=True,padding_side='left')
devices=dict.fromkeys(('mulacover','qwen','codec','transcriptor'),torch.device('cuda'))
pipe=MuLaCoverGenPipeline({'mulacover':g,'qwen':s},devices,dict.fromkeys(devices,torch.float32),text,MuLaCoverGenConfig.from_file(g/'gen_config.json'),MuLaCoverConfig.from_pretrained(g),lazy_load=True)
cases=[('original','[Verse]\nA lantern swings beside the rain\nWe count the lights along the train\n',' Warm Acoustic Pop, Soft Drums, Clear Voice '),
       ('instrumental','[Instrumental]','warm acoustic pop, soft drums, instrumental, no vocals'),
       ('unicode','[Verse]\nCAFÉ cafe\u0301 Ελληνικά ΟΣ ΟΣΑ İSTANBUL 音楽 １２３４ 😀',' <TAG>  Café, ΑΚΟΥΣΤΙΚΉ, soft\tDRUMS </TAG> '),
       ('empty-style','[Instrumental]','  '),
       ('specials',"[VERSE]\nI'M I'LL we're don't 1234567 <|end_of_text|>\r\n",'<tag> pop <|endoftext|> </tag>')]
manifest={'upstream_revision':upstream_revision,'style_revision':sources['style']['revision'],'dtype':'float32','attention':'eager','versions':{x:importlib.metadata.version(x) for x in ['torch','transformers','tokenizers','numpy']},'cases':[],'style_weight_tensors':len(state)+2}
for name,lyrics,tags in cases:
    root=a.output/name;root.mkdir(exist_ok=True)
    (root/'lyrics.txt').write_text(lyrics,encoding='utf-8');(root/'tags.txt').write_text(tags,encoding='utf-8')
    lyric=lyrics.lower();tag=tags.strip().lower()
    if tag.startswith('<tag>') and tag.endswith('</tag>'):tag=tag[5:-6].strip()
    (root/'normalized-lyrics.txt').write_text(lyric,encoding='utf-8');(root/'normalized-tags.txt').write_text(tag,encoding='utf-8')
    tokens,mask,qi,mi=pipe._build_prompt(lyric,tag)
    def save(name,value,dtype=None):
        if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
        np.ascontiguousarray(value,dtype=dtype).tofile(root/name)
    save('tokens.i32',tokens,np.int32);save('mask.u8',mask,np.uint8)
    batch=qtok([tag],return_tensors='pt',padding=True,truncation=False)
    ids=batch['input_ids'][0];save('style-ids.i32',ids,np.int32)
    with torch.inference_mode():
        if tag:
            out=model(**{k:v.cuda() for k,v in batch.items()},output_hidden_states=True)
            hidden=out.last_hidden_state;style=torch.nn.functional.normalize(hidden[:,-1].float(),dim=-1)[0]
            save('style-hidden.f32',hidden)
            for index in [0,1,14,28]:save(f'style-layer-{index}.f32',out.hidden_states[index])
        else:style=torch.zeros(1024)
    save('qwen.f32',style);save('muq.f32',np.zeros(512,np.float32))
    manifest['cases'].append({'name':name,'prompt_frames':len(tokens),'qwen_index':qi,'muq_index':mi,'style_tokens':len(ids),'style_skipped':not tag})
    print('Frozen',name,len(tokens),len(ids),flush=True)
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Wrote tokenizer assets and F32 style weights',flush=True)
