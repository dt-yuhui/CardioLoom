# Text encoder files

Text-encoder weights are not stored in this repository.

By default, pretraining uses the Hugging Face model ID
`ncbi/MedCPT-Query-Encoder`. On a machine with internet access, Transformers
downloads and caches it automatically.

For an offline server, download the complete model repository on another
machine, transfer the resulting directory here (or to any other local path),
and run pretraining with:

```bash
--text_model pretrained_models/text_encoder/ncbi/MedCPT-Query-Encoder \
--text_model_local_files_only True
```

The `.gitignore` intentionally excludes all weight files below this directory
while keeping this README as a placeholder.

