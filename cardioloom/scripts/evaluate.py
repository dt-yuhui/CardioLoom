"""Single-endpoint paired evaluation on aligned, privately held local scores."""
import argparse
import json
import numpy as np
import pandas as pd
from cardioloom.models.corrected_svm import search_best_raw_threshold
from cardioloom.statistics import paired_bootstrap, bh_adjust


def compare(val, test, seed=20260905):
    models=['reference','comparator']
    for df in (val,test):
        if not df.label.isin([0,1]).all() or not np.isfinite(df[models]).all().all(): raise ValueError('Require binary labels and finite paired scores')
    if test.cluster.isna().any(): raise ValueError('Missing cluster ID')
    y=test.label.to_numpy(); obs, draws=paired_bootstrap(y,[test[m] for m in models],test.cluster.to_numpy(),repeats=1000,seed=seed); valid=np.isfinite(draws).all(axis=(1,2)); res={}
    for i,m in enumerate(models):
        threshold,_=search_best_raw_threshold(val.label.to_numpy(),val[m].to_numpy()); pred=test[m].to_numpy()>=threshold
        res[m]={'validation_threshold':threshold,'TP':int(sum(pred&(y==1))),'FP':int(sum(pred&(y==0))),'TN':int(sum(~pred&(y==0))),'FN':int(sum(~pred&(y==1))),'AUPRC':obs[i,0],'AUROC':obs[i,1]}
    for i,metric in enumerate(['AUPRC','AUROC']):
        delta=obs[0,i]-obs[1,i]; z=draws[valid,0,i]-draws[valid,1,i]; ok=len(z)>=950
        res['difference_'+metric]={'estimate':delta,'valid_draws':len(z),'ci95':np.quantile(z,[.025,.975]).tolist() if ok else None,'centered_p':float((1+sum(abs(z-delta)>=abs(delta)))/(len(z)+1)) if ok else None}
    return res


def main():
    p=argparse.ArgumentParser(description=__doc__); s=p.add_subparsers(dest='cmd',required=True); v=s.add_parser('compare'); v.add_argument('validation'); v.add_argument('test'); v.add_argument('--seed',type=int,default=20260905); b=s.add_parser('bh'); b.add_argument('family_csv'); a=p.parse_args()
    if a.cmd=='compare': print(json.dumps(compare(pd.read_csv(a.validation),pd.read_csv(a.test,dtype={'cluster':str}),a.seed),indent=2,allow_nan=False))
    else:
        df=pd.read_csv(a.family_csv); df['q']=np.nan
        if df.family.isna().any() or not df.p.dropna().between(0,1).all(): raise ValueError('Invalid family or p')
        for _,g in df.dropna(subset=['p']).groupby('family'): df.loc[g.index,'q']=bh_adjust(g.p)
        print(df.to_csv(index=False),end='')
if __name__=='__main__': main()
