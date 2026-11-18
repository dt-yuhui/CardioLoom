"""Shared stdlib-only argument and safety helpers for dry-run-capable entry points."""

from __future__ import annotations

import argparse
import __future__ as future_features
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
import os
CODEX_ROOT = Path(os.environ.get('CARDIOLOOM_WORK', PROJECT_ROOT / 'work')).resolve()
PUBLIC_SOURCE = PROJECT_ROOT / 'cardioloom/runtime'
DEFAULT_DATA_ROOT = Path(os.environ.get('DATA_ROOT', PROJECT_ROOT / 'data'))
DEFAULT_DEEPECG_CKPT = Path(os.environ.get('DEEPECG_CHECKPOINT', 'weights/SSL_pretrained.pt'))
DEFAULT_ECGCLIP_DEEPECG_CKPT = Path(os.environ.get('CARDIOLOOM_CHECKPOINT', 'weights/CardioLoom.pt'))
DEFAULT_DEEPECG_FAIRSEQ = Path(os.environ.get('DEEPECG_FAIRSEQ', 'third_party/DeepECG-SSL-finetune'))
ECGCLIP_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
DEFAULT_EXCLUDE_LABEL_IDS = (
    5, 10, 12, 20, 22, 23, 25, 26, 30, 35, 37, 50, 54, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 69, 70, 71, 72, 73, 74,
)
ECHO_EXCLUDE_LABEL_IDS = (8, 9, 14)


def prepare_public_source_imports() -> None:
    """Load utils.util with postponed annotations for Python 3.9 compatibility."""
    public_source = str(PUBLIC_SOURCE)
    if public_source not in sys.path:
        sys.path.insert(0, public_source)

    module_name = "utils.util"
    if module_name in sys.modules:
        return

    import utils

    source_path = PUBLIC_SOURCE / "utils/util.py"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None:
        raise ImportError(f"Could not create an import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        code = compile(
            source_path.read_bytes(),
            str(source_path),
            "exec",
            flags=future_features.annotations.compiler_flag,
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        setattr(utils, "util", module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def bool_value(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        choices=("ecg_clip_deepecg",),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--deepecg-base-checkpoint",
        type=Path,
        default=DEFAULT_DEEPECG_CKPT,
        help=(
            "DeepECG checkpoint used to initialize the CardioLoom backbone."
        ),
    )
    parser.add_argument("--fairseq-root", type=Path)
    parser.add_argument("--source-lead-order", type=comma_strings, default=ECGCLIP_LEADS)
    parser.add_argument(
        "--deepecg-normalization-mode",
        choices=("fft_power_factor", "mv_scale"),
        default=None,
        help="Required for DeepECG-SSL; select the explicitly justified official normalization family.",
    )
    parser.add_argument(
        "--deepecg-input-scale",
        type=float,
        default=None,
        help="Required for DeepECG-SSL; FFT power factor or raw-unit-to-mV factor, as selected above.",
    )


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--h5-file-path", default="ecg_data_all.h5")
    parser.add_argument("--label-file-path", default="ecg_labels_all.csv")
    parser.add_argument("--info-file-path", default="all_information_v8.csv")
    parser.add_argument("--train-indices-file-path", default="new_split_v2/downstream_train_indices.csv")
    parser.add_argument("--valid-indices-file-path", default="new_split_v2/downstream_valid_indices.csv")
    parser.add_argument("--test-indices-file-path", default="new_split_v2/downstream_test_indices.csv")
    parser.add_argument("--downstream-task", required=True)
    parser.add_argument("--ecg-or-echo", default="ecg", choices=("ecg", "echo"))
    parser.add_argument("--exclude-label-ids", type=comma_ints, default=DEFAULT_EXCLUDE_LABEL_IDS)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--random-split-seed", type=int, default=None)


def resolve_model_paths(args: argparse.Namespace) -> None:
    args.checkpoint = args.checkpoint or DEFAULT_ECGCLIP_DEEPECG_CKPT
    args.fairseq_root = args.fairseq_root or DEFAULT_DEEPECG_FAIRSEQ
    if args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint).resolve()
    args.fairseq_root = Path(args.fairseq_root).resolve()
    args.deepecg_base_checkpoint = Path(args.deepecg_base_checkpoint).resolve()
    if tuple(args.source_lead_order) != ECGCLIP_LEADS:
        raise ValueError(f"source lead order must be exactly {ECGCLIP_LEADS}")
    if args.model in {"deepecg_ssl", "ecg_clip_deepecg", "deepecg_scratch"}:
        if args.deepecg_normalization_mode is None:
            raise ValueError("DeepECG-SSL requires --deepecg-normalization-mode")
        if args.deepecg_input_scale is None or args.deepecg_input_scale <= 0:
            raise ValueError("DeepECG-SSL requires a positive --deepecg-input-scale")


def assert_under_codex(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != CODEX_ROOT.resolve() and CODEX_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"Output must be under {CODEX_ROOT}; got {resolved}")
    return resolved


def assert_new_output(path: Path) -> Path:
    resolved = assert_under_codex(path)
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {resolved}")
    return resolved


def known_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.data_root.resolve()
    return {
        "h5": root / args.h5_file_path,
        "labels": root / args.label_file_path,
        "info": root / args.info_file_path,
        "train_split": root / args.train_indices_file_path,
        "valid_split": root / args.valid_indices_file_path,
        "test_split": root / args.test_indices_file_path,
    }


def dry_run_report(args: argparse.Namespace, *, output_paths: dict[str, Path]) -> None:
    resolve_model_paths(args) if hasattr(args, "model") else None
    report: dict[str, Any] = {
        "mode": "dry_run",
        "project_source": str(PUBLIC_SOURCE),
        "inputs": {name: {"path": str(path), "exists": path.exists()} for name, path in known_input_paths(args).items()},
        "outputs": {name: str(assert_under_codex(path)) for name, path in output_paths.items()},
        "dataset_contract": {
            "tensor_shape": "[B,8,5000]",
            "sampling_rate_hz": 500,
            "lead_order": ECGCLIP_LEADS,
            "labels_and_splits": "original ECGDatasetDownstream paths; no files opened in dry-run",
        },
    }
    if hasattr(args, "model"):
        report["model"] = {
            "name": args.model,
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "checkpoint_exists": args.checkpoint.is_file() if args.checkpoint is not None else False,
            "checkpoint_bytes": (
                args.checkpoint.stat().st_size
                if args.checkpoint is not None and args.checkpoint.is_file()
                else None
            ),
            "initialization": "checkpoint",
            "fairseq_root": str(args.fairseq_root),
            "fairseq_exists": (args.fairseq_root / "fairseq_signals").is_dir(),
            "normalization_mode": args.deepecg_normalization_mode,
            "input_scale": args.deepecg_input_scale,
            "base_checkpoint": str(args.deepecg_base_checkpoint),
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_adapter(args: argparse.Namespace):
    resolve_model_paths(args)
    from cardioloom.models.ecgclip_deepecg_adapter import ECGCLIPDeepECGAdapter
    return ECGCLIPDeepECGAdapter(
        checkpoint_path=args.checkpoint, base_checkpoint_path=args.deepecg_base_checkpoint,
        fairseq_root=args.fairseq_root, source_lead_order=args.source_lead_order,
        input_scale=args.deepecg_input_scale, normalization_mode=args.deepecg_normalization_mode,
    )

def original_downstream_namespace(args: argparse.Namespace, output_dir: Path) -> SimpleNamespace:
    """Build exactly the attribute surface consumed by original downstream scripts."""
    return SimpleNamespace(
        seed=args.seed,
        split_random_state=42,
        exp_name=args.exp_name,
        log_name=f"internal_{getattr(args, 'model', 'svm')}",
        resume=None,
        in_features=768,
        num_classes=args.num_classes,
        finetune_type=getattr(args, "finetune_type", "finetune"),
        cuda=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        downstream_task=args.downstream_task,
        test_only=args.test_only,
        exclude_label_ids=args.exclude_label_ids,
        rare_disease_label_ids=(),
        ecg_or_echo=args.ecg_or_echo,
        data_root=str(args.data_root.resolve()),
        h5_file_path=args.h5_file_path,
        label_file_path=args.label_file_path,
        info_file_path=args.info_file_path,
        train_indices_file_path=args.train_indices_file_path,
        valid_indices_file_path=args.valid_indices_file_path,
        test_indices_file_path=args.test_indices_file_path,
        train_offline_features_file=getattr(args, "train_offline_features_file", None),
        valid_offline_features_file=getattr(args, "valid_offline_features_file", None),
        test_offline_features_file=getattr(args, "test_offline_features_file", None),
        sample_ratio=args.sample_ratio,
        random_split_seed=args.random_split_seed,
        external_dataset="",
        num_leads=8,
        downsample=False,
        ecg_model="ecgfounder",
        custom_pretrained_root=str(CODEX_ROOT / "checkpoints"),
        pretrained_exp_name=getattr(args, "model", "offline_features"),
        pretrained_metric=None,
        ckpt_name="",
        drop_path=0.0,
        lr=getattr(args, "lr", 1e-4),
        weight_decay=getattr(args, "weight_decay", 5e-2),
        epochs=getattr(args, "epochs", 100),
        saveckp_freq=1,
        val_iter=getattr(args, "val_iter", 200),
        total_iterations=getattr(args, "total_iterations", 4000),
        experiment_path=str(output_dir),
        tfboard_path=str(output_dir / "tfboard"),
        ckpt_path=str(output_dir / "best-checkpoint.pth"),
    )
