"""Dispatch named paper runs using the archived configuration values."""
import argparse
import json
import os
import sys
from pathlib import Path
from cardioloom.scripts.common import CODEX_ROOT, PROJECT_ROOT, DEFAULT_DEEPECG_FAIRSEQ

SPARSE = {'PR','PS','TS', *['echo_label_'+str(i) for i in (8,9,14)], *['ecg_label_'+str(i) for i in (63,65,71,73,74)]}

def model_args():
    return ['--model','ecg_clip_deepecg','--deepecg-normalization-mode','mv_scale','--deepecg-input-scale','1']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['pretrain','finetune','features','svm'])
    p.add_argument('task',nargs='?')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); model="cardioloom"; task=a.task; stage=a.stage
    if stage in ('finetune','svm') and not task: p.error('This stage requires a task')
    if stage in ('pretrain','features') and task: p.error('Unexpected task')
    opts=['--exp-name',model+(('_'+task) if task else '')]
    launcher=[]
    if stage=='pretrain':
        opts += ['--fairseq-root',str(DEFAULT_DEEPECG_FAIRSEQ),'--text-model',os.getenv('MEDCPT_DIR','weights/MedCPT-Query-Encoder'),'--ecg-backbone-lr','5e-6','--freeze-ecg-backbone-updates','500','--report-sampling-alpha','0.25','--cma-loss','multi_positive']
        if not a.dry_run: launcher=['-m','torch.distributed.run','--standalone','--nproc_per_node=2','--module']
    elif stage=='finetune':
        cfg=json.loads((PROJECT_ROOT/'configs/training.json').read_text()); key=task
        if key not in cfg: p.error('No archived fine-tuning recipe for '+key)
        opts += model_args()
        for k,v in cfg[key].items(): opts += ['--'+k.replace('_','-'),str(v)]
    elif stage=='features':
        opts += model_args()+['--downstream-task','ecg_multi_label','--feature-selection','all_split']
    else:
        if task not in SPARSE: p.error('Choose a pretrained encoder and a supported sparse task')
        opts += ['--downstream-task',task,'--ecg-or-echo','ecg' if task.startswith('ecg_') else 'echo','--svm-protocol','rawscore_v2']
        for split in ['train','valid','test']:
            opts += ['--'+split+'-offline-features-file',str(CODEX_ROOT/'outputs/features'/model/(split+'_features.h5'))]
    if a.dry_run: opts.append('--dry-run')
    os.execv(sys.executable,[sys.executable]+(launcher or ['-m'])+['cardioloom.scripts.'+stage]+opts)
if __name__=='__main__': main()
