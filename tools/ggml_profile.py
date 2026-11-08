"""Offline validation provenance for the coordinated shared GGML dependency."""
import json
import os
from pathlib import Path
import subprocess

PROFILE = json.loads((Path(__file__).resolve().parents[1] / 'dependencies/ggml.json').read_text())
PIN = PROFILE['revision']

def validate_checkout(source):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True,env=env).strip()
    dirty=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True,env=env).strip()
    if revision!=PIN or dirty:
        raise ValueError(f'Validation requires the clean coordinated GGML pin {PIN}; got {revision}, dirty={bool(dirty)}')
    return revision
