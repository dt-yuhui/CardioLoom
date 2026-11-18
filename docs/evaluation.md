# Inference and evaluation

For external inference, convert local waveforms to the canonical HDF5 contract described in `reproduction.md`. Apply the same upstream waveform processing as for the cohort of interest. `cardioloom/external_io.py` contains optional reader helpers. Hospital-specific linkage, cohort inclusion and endpoint adjudication are not reproduced by the generic exporter. Supply only trusted local checkpoints. No patient identifiers need to leave your site.

```bash
python -m cardioloom.scripts.predict --input external.h5 --head head.pt --output scores.npz --device cuda
# Alternative for a sparse target with a separately fitted raw-margin classifier:
python -m cardioloom.scripts.predict --input external.h5 --svm PS.pkl --output PS_scores.npz --device cuda
```

Set the upstream and pretrained encoder paths first, even when loading an end-to-end head. The output `scores` matrix follows HDF5 row order and the accompanying `tasks` array identifies its columns. End-to-end heads use positive-class softmax probabilities. Sparse SVMs use signed decision margins. Use heads and pickle files produced by this distribution; older pickles may refer to a private Python module path. Keep individual scores private.

For one endpoint, prepare a local CSV per split with columns `label,score` in the same row order as the corresponding records. Use eligible binary labels only. The following commands report trapezoidal AUPRC, AUROC and operating-point metrics. Thresholds are chosen on validation and applied unchanged to test. For the CA screening analysis select the high-specificity rule:

```bash
python -m cardioloom.scripts.metrics validation.csv test.csv
python -m cardioloom.scripts.metrics ca_validation.csv ca_test.csv --rule spec99
```

To report a macro-AUPRC, take the unweighted mean over the prespecified target set for that analysis. Report nonestimable endpoints separately. For the routine ECG macro, use the 45 routine diagnoses in the task dictionary, not every column of the multitask head.
