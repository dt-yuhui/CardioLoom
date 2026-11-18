"""Export row-aligned scores on a canonical external HDF5, using a trusted local head."""
import argparse
import pickle
from pathlib import Path
import h5py
import numpy as np
import torch
from cardioloom.launch import model_args
from cardioloom.scripts.common import add_model_args, resolve_model_paths, build_adapter
from cardioloom.models.corrected_svm import assert_rawscore_checkpoint
from cardioloom.models.finetuner import FoundationFineTuner

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument('--head',type=Path); g.add_argument('--svm',type=Path)
    p.add_argument('--input',required=True,type=Path); p.add_argument('--output',required=True,type=Path); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--device',default='cpu',choices=['cuda','cpu']); a=p.parse_args()
    if a.output.exists() or a.batch_size<1: p.error('Use a new output file and positive batch size')
    backend=argparse.ArgumentParser(); add_model_args(backend); args=backend.parse_args(model_args()); resolve_model_paths(args)
    adapter=build_adapter(args)
    if a.head:
        saved=torch.load(a.head,map_location='cpu'); tasks=list(saved['task_names'] if 'task_names' in saved else saved['threshold']); net=FoundationFineTuner(adapter,adapter.feature_dim,0.,dict.fromkeys(tasks,2)); net.load_state_dict(saved['model']); net.to(a.device).eval()
    else:
        with a.svm.open('rb') as f: saved=pickle.load(f)
        assert_rawscore_checkpoint(saved); tasks=list(saved['threshold']); assert len(tasks)==1; net=adapter.to(a.device).eval()
    blocks=[]
    with h5py.File(a.input,'r') as f, torch.inference_mode():
        data=f['ECGs']; n=len(data)
        if data.shape!=(n,8,5000) or not n: raise ValueError('Expected nonempty [N,8,5000] ECGs in mV')
        for start in range(0,n,a.batch_size):
            x=torch.as_tensor(np.asarray(data[start:start+a.batch_size]),dtype=torch.float32,device=a.device)
            if not torch.isfinite(x).all(): raise ValueError('Nonfinite input')
            if a.head:
                logits=net(x); scores=torch.stack([logits[t].float().softmax(-1)[:,1] for t in tasks],dim=1).cpu().numpy()
            else: scores=saved['model'].decision_function(net.forward_features(x).cpu().numpy()).reshape(-1,1)
            if not np.isfinite(scores).all(): raise ValueError('Nonfinite score')
            blocks.append(scores)
    with a.output.open('xb') as f: np.savez_compressed(f,tasks=np.asarray(tasks),scores=np.concatenate(blocks))
if __name__=='__main__': main()
