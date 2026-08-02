# ECG-CLIP Repository

[![arXiv](https://img.shields.io/badge/arXiv-2605.25446-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2605.25446)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

This repository hosts two independent ECG research projects. They are kept in
separate directories and do not depend on each other.

| Project | Description | Documentation |
|:--|:--|:--:|
| **ECG-CLIP** | Signal-language foundation-model pretraining and downstream evaluation for broad-spectrum cardiovascular assessment. | [Open project](ECG-CLIP/README.md) |
| **VHD-Survival** | A separate project for ECG-based prognosis modeling in valvular heart disease. | [Open project](VHD-Survival/README.md) |

## ECG-CLIP

The [`ECG-CLIP/`](ECG-CLIP/) directory contains the official source-code
release for:

> **A Signal-Language Foundation Model for Broad-Spectrum Cardiovascular Assessment from Routine Electrocardiography**

- Paper: [arXiv:2605.25446](https://arxiv.org/abs/2605.25446)
- Included: contrastive pretraining, ECGCLIP-R34 fine-tuning, internal
  evaluation, and rare-disease SVM evaluation.
- Planned: author-pretrained ECGCLIP-R18, ECGCLIP-R34, and ECGCLIP-R50
  checkpoints will be released after paper acceptance.

For installation, data conventions, model release status, and experiment
instructions, see the [ECG-CLIP documentation](ECG-CLIP/README.md).

## VHD-Survival

The existing [`VHD-Survival/`](VHD-Survival/) directory is retained as an
independent project. Its training and prediction instructions are available in
the [VHD-Survival documentation](VHD-Survival/README.md).

## License

The repository-level license is available in [LICENSE](LICENSE). Individual
subdirectories may also contain project-specific notices.
