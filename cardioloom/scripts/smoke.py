"""CPU-only functional check using artificial signals, reports and labels."""
import tempfile
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from cardioloom.pretrain_dataset import ReadOnlyECGPretrainDataset
from cardioloom.multi_positive_clip import multi_positive_clip_loss
from cardioloom.models.ecg_preprocessing import reconstruct_standard_12_leads
from cardioloom.scripts.evaluate import compare
from cardioloom.statistics import bh_adjust

def main():
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp); rng=np.random.default_rng(42); x=rng.normal(size=(8,8,5000)).astype('f4'); ids=['synthetic-'+str(i) for i in range(8)]
        with h5py.File(p/'waves.h5','w') as f: f['ECGs']=x
        pd.DataFrame({'Filename':ids}).to_csv(p/'labels.csv',index=False); pd.DataFrame({'Filename':ids,'english_report':['template a','template a','template b','template c','template d','template e','template f','template g']}).to_csv(p/'reports.csv',index=False)
        for split,rows in [('train',range(6)),('valid',range(6,8))]: pd.DataFrame({'index':rows}).to_csv(p/(split+'.csv'),index=False)
        ds,val=ReadOnlyECGPretrainDataset.create_datasets(p/'waves.h5',p/'labels.csv',p/'reports.csv',p/'train.csv',p/'valid.csv',8,include_report_template_ids=True)
        assert len(ds)==6 and len(val)==2 and torch.equal(ds[0]['ecg_data'],torch.from_numpy(x[0])); w=ds.get_report_sampling_weights(.25); assert abs(float(w[0]/w[2])-2**(-.25))<1e-8
        z=reconstruct_standard_12_leads(torch.from_numpy(x)); torch.testing.assert_close(z[:,2],z[:,1]-z[:,0]); torch.manual_seed(42)
        u=torch.randn(6,16,requires_grad=True); loss,_,_,diag=multi_positive_clip_loss(u,u,ds.report_template_ids,temperature=.07); loss.backward(); assert torch.isfinite(u.grad).all() and int(diag['report_positive_count_max'])==2
        ds._close_h5(); val._close_h5()
    y=np.tile([0,1],50); d=pd.DataFrame({'label':y,'reference':y*.8+.1,'comparator':1-y,'cluster':np.repeat(np.arange(50),2)}); r=compare(d,d); assert r['difference_AUROC']['estimate']==1 and r['reference']['FP']==0 and r['difference_AUPRC']['valid_draws']==1000; np.testing.assert_allclose(bh_adjust([.03,.001]),[.03,.002]); print('PASS: synthetic input, lead reconstruction, template sampling, multi-positive loss/backward, validation threshold, paired bootstrap and BH.')
if __name__=='__main__': main()
