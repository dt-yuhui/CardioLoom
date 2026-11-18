"""Pretrain 8/12-lead Transformer ECGCLIP with the original ECGCLIP loop."""
from __future__ import annotations
import argparse
import atexit
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cardioloom.models.ecgclip_transformer import ECGCLIPTransformer, SyntheticTextEncoder
from cardioloom.multi_positive_clip import distributed_concat_ids, multi_positive_clip_loss
from cardioloom.pretrain_dataset import ReadOnlyECGPretrainDataset
from cardioloom.report_sampling import DistributedTemplateWeightedSampler
from cardioloom.scripts.common import CODEX_ROOT, DEFAULT_DATA_ROOT, DEFAULT_DEEPECG_CKPT, DEFAULT_DEEPECG_FAIRSEQ, PUBLIC_SOURCE, bool_value, prepare_public_source_imports
DEFAULT_TEXT_MODEL = Path(os.environ.get('MEDCPT_DIR', 'weights/MedCPT-Query-Encoder'))
DEFAULT_FAIRSEQ_ROOT = DEFAULT_DEEPECG_FAIRSEQ
LEAD_ORDERS = {8: ('I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'), 12: ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--output-root', type=Path, default=CODEX_ROOT / 'outputs/pretrain')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--h5-file-path', default='ecg_data_all.h5')
    parser.add_argument('--label-file-path', default='ecg_labels_all.csv')
    parser.add_argument('--info-file-path', default='all_information.csv')
    parser.add_argument('--train-indices-file-path', default='new_split_v2/pretrain_train_indices.csv')
    parser.add_argument('--valid-indices-file-path', default='new_split_v2/pretrain_valid_indices.csv')
    parser.add_argument('--text-model', type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument('--fairseq-root', type=Path, default=DEFAULT_FAIRSEQ_ROOT)
    parser.add_argument('--backbone-init', choices=('deepecg_ssl',), default='deepecg_ssl', help='Initialize the ECG encoder from DeepECG-SSL.')
    parser.add_argument('--deepecg-checkpoint', type=Path, default=DEFAULT_DEEPECG_CKPT)
    parser.add_argument('--deepecg-normalization-mode', choices=('mv_scale', 'fft_power_factor'), default='mv_scale')
    parser.add_argument('--deepecg-input-scale', type=float, default=1.0)
    parser.add_argument('--num-leads', type=int, choices=(8, 12), default=8)
    parser.add_argument('--conv-channels', type=int, default=256)
    parser.add_argument('--conv-layers', type=int, default=4)
    parser.add_argument('--encoder-dim', type=int, default=768)
    parser.add_argument('--encoder-layers', type=int, default=12)
    parser.add_argument('--encoder-heads', type=int, default=12)
    parser.add_argument('--encoder-ffn-dim', type=int, default=3072)
    parser.add_argument('--encoder-dropout', type=float, default=0.1)
    parser.add_argument('--attention-dropout', type=float, default=0.1)
    parser.add_argument('--activation-dropout', type=float, default=0.0)
    parser.add_argument('--dropout-input', type=float, default=0.1)
    parser.add_argument('--feature-grad-mult', type=float, default=0.1)
    parser.add_argument('--positional-kernel', type=int, default=128)
    parser.add_argument('--positional-groups', type=int, default=16)
    parser.add_argument('--gradient-checkpointing', type=bool_value, default=True)
    parser.add_argument('--mlp-hidden-size', type=int, default=256)
    parser.add_argument('--projection-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train-batch-size-per-gpu', type=int, default=700)
    parser.add_argument('--valid-batch-size-per-gpu', type=int, default=768)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--ecg-backbone-lr', type=float, default=None, help='Optional differential LR for ecg_encoder; --lr remains the projection/text LR.')
    parser.add_argument('--text-encoder-lr', type=float, default=None, help='Optional LR for trainable lm_model parameters. When set, --lr is reserved for randomly initialized projection/UMA heads.')
    parser.add_argument('--freeze-ecg-backbone-updates', type=int, default=0, help='Train projection/text parameters only for the first N optimizer updates.')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.98)
    parser.add_argument('--adam-eps', type=float, default=1e-06)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-updates', type=int, default=500, help='Linear LR warmup updates before cosine warm restarts.')
    parser.add_argument('--max-grad-norm', type=float, default=5.0, help='Global gradient L2-norm clipping threshold; 0 disables clipping.')
    parser.add_argument('--scheduler-reference-t0', type=int, default=5000)
    parser.add_argument('--scheduler-reference-global-batch', type=int, default=1400)
    parser.add_argument('--scheduler-t0', type=int, default=None)
    parser.add_argument('--lr-scheduler-type', choices=('warm_restarts', 'single_cosine'), default='warm_restarts', help='Preserve historical restarts or use one warmup-plus-cosine trajectory.')
    parser.add_argument('--scheduler-min-lr-ratio', type=float, default=0.1, help="For single_cosine, final LR as a ratio of each parameter group's peak LR.")
    parser.add_argument('--uma-weight', type=float, default=1.0)

    parser.add_argument('--save-encoder-every-epoch', type=bool_value, default=False, help='Save an ECG-encoder-only checkpoint after every completed epoch.')
    parser.add_argument('--cma-loss', choices=('multi_positive',), default='multi_positive', help='Treat ECGs paired with the same normalized report as positives.')
    parser.add_argument('--report-sampling-alpha', type=float, default=0.25, help='Per-row weight is template_frequency**(-alpha); 0 preserves uniform rows.')
    parser.add_argument('--contrastive-temperature', type=float, default=0.07)

    parser.add_argument('--early-stopping-patience', type=int, default=0, help='Epochs without min-delta improvement; 0 disables early stopping.')
    parser.add_argument('--early-stopping-min-epochs', type=int, default=0)
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0)
    parser.add_argument('--amp-dtype', choices=('bf16', 'fp32'), default='bf16', help='Training/validation precision. BF16 matches the historical ECGCLIP loop.')
    parser.add_argument('--detect-anomaly', type=bool_value, default=False, help='Enable expensive autograd anomaly detection for debugging only.')
    parser.add_argument('--freeze-text-encoder', type=bool_value, default=False)
    parser.add_argument('--text-encoder-freeze-layers', type=int, default=9)
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dist-url', default='env://')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--log-name', default='ECG_CLIP_TRANSFORMER')
    parser.add_argument('--monitor-interval', type=int, default=50, help='Write aggregate Transformer/optimizer diagnostics every N train steps')
    parser.add_argument('--log-gradient-norms', type=bool_value, default=True)
    parser.add_argument('--dry-run', action='store_true')
    return parser

def _resolved_output(args: argparse.Namespace) -> tuple[Path, Path]:
    output_root = args.output_root.resolve()
    codex_root = CODEX_ROOT.resolve()
    if output_root != codex_root and codex_root not in output_root.parents:
        raise ValueError(f'output-root must be under {codex_root}: {output_root}')
    return (output_root / args.exp_name, output_root / 'tfboard' / args.exp_name)

def _validate_args(args: argparse.Namespace) -> None:
    expected_leads = {'scratch8': 8, 'scratch12': 12, 'deepecg_ssl': 8}[args.backbone_init]
    if args.num_leads != expected_leads:
        raise ValueError(f'backbone-init={args.backbone_init} requires num-leads={expected_leads}, got {args.num_leads}')
    if args.encoder_dim % args.encoder_heads:
        raise ValueError('encoder-dim must be divisible by encoder-heads')
    if args.encoder_dim % args.positional_groups:
        raise ValueError('encoder-dim must be divisible by positional-groups')
    if args.train_batch_size_per_gpu < 5 or args.valid_batch_size_per_gpu < 5:
        raise ValueError('CLIP top-5 accuracy requires train/valid batch sizes >= 5')
    if args.monitor_interval <= 0:
        raise ValueError('monitor-interval must be positive')
    if not args.fairseq_root.is_dir():
        raise FileNotFoundError(args.fairseq_root)
    if args.scheduler_reference_t0 <= 0 or args.scheduler_reference_global_batch <= 0:
        raise ValueError('scheduler reference values must be positive')
    if args.scheduler_t0 is not None and args.scheduler_t0 <= 0:
        raise ValueError('scheduler-t0 must be positive')
    if args.lr <= 0:
        raise ValueError('lr must be positive')
    if args.ecg_backbone_lr is not None and args.ecg_backbone_lr <= 0:
        raise ValueError('ecg-backbone-lr must be positive when set')
    if args.text_encoder_lr is not None and args.text_encoder_lr <= 0:
        raise ValueError('text-encoder-lr must be positive when set')
    if args.freeze_ecg_backbone_updates < 0:
        raise ValueError('freeze-ecg-backbone-updates must be non-negative')
    if not 0 <= args.momentum < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError('Adam betas must be in [0, 1)')
    if args.adam_eps <= 0:
        raise ValueError('adam-eps must be positive')
    if args.weight_decay < 0:
        raise ValueError('weight-decay must be non-negative')
    if args.warmup_updates < 0:
        raise ValueError('warmup-updates must be non-negative')
    if not 0.0 <= args.scheduler_min_lr_ratio < 1.0:
        raise ValueError('scheduler-min-lr-ratio must be in [0,1)')
    if args.max_grad_norm < 0:
        raise ValueError('max-grad-norm must be non-negative')
    if not 0.0 <= args.report_sampling_alpha <= 1.0:
        raise ValueError('report-sampling-alpha must be in [0,1]')
    if args.contrastive_temperature <= 0:
        raise ValueError('contrastive-temperature must be positive')
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 0:
        raise ValueError('early-stopping patience/min-epochs must be non-negative')
    if args.early_stopping_min_delta < 0:
        raise ValueError('early-stopping-min-delta must be non-negative')
    if args.early_stopping_patience > 0:
        if not 1 <= args.early_stopping_min_epochs <= args.epochs:
            raise ValueError('Enabled early stopping requires min-epochs in [1,epochs]')
    elif args.early_stopping_min_epochs != 0 or args.early_stopping_min_delta != 0.0:
        raise ValueError('early-stopping min-epochs/min-delta require positive patience')
    if args.cma_loss == 'diagonal' and (not math.isclose(args.contrastive_temperature, 0.07, rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError('Historical diagonal CMA has a fixed temperature of 0.07; do not provide a different contrastive-temperature')
    if args.resume is not None and (not args.resume.is_file()):
        raise FileNotFoundError(args.resume)
    if args.backbone_init == 'deepecg_ssl':
        if not args.deepecg_checkpoint.is_file():
            raise FileNotFoundError(args.deepecg_checkpoint)
        if args.fairseq_root.resolve() != DEFAULT_DEEPECG_FAIRSEQ.resolve():
            if not (args.fairseq_root / 'fairseq_signals').is_dir():
                raise FileNotFoundError(args.fairseq_root / 'fairseq_signals')
        if args.deepecg_input_scale <= 0:
            raise ValueError('deepecg-input-scale must be positive')

def _build_model(args: argparse.Namespace, *, text_encoder: nn.Module | None=None, tokenizer=None, load_text_encoder: bool=True) -> nn.Module:
    if args.backbone_init == 'deepecg_ssl':
        from cardioloom.models.ecgclip_deepecg import ECGCLIPDeepECG
        return ECGCLIPDeepECG(args, text_encoder=text_encoder, tokenizer=tokenizer, load_text_encoder=load_text_encoder)
    return ECGCLIPTransformer(args, text_encoder=text_encoder, tokenizer=tokenizer, load_text_encoder=load_text_encoder)

def _dry_run(args):
    print(json.dumps(vars(args), default=str, indent=2))

def _formal_args(args: argparse.Namespace) -> argparse.Namespace:
    (experiment_path, tfboard_path) = _resolved_output(args)
    if args.resume is not None and args.resume.resolve().parent != experiment_path:
        raise ValueError('Resume checkpoint must belong to the current experiment directory; cross-experiment continual pretraining requires a new explicit initialization path, not --resume')
    global_rank = int(os.environ.get('RANK', '0'))
    if global_rank == 0 and args.resume is None:
        occupied = [path for path in (experiment_path, tfboard_path) if path.is_dir() and next(path.iterdir(), None) is not None]
        if occupied:
            joined = ', '.join((str(path) for path in occupied))
            raise FileExistsError(f'Refusing to overwrite non-empty output: {joined}')
    experiment_path.mkdir(parents=True, exist_ok=True)
    tfboard_path.mkdir(parents=True, exist_ok=True)
    args.experiment_path = str(experiment_path)
    args.tfboard_path = str(tfboard_path)
    args.data_root = str(args.data_root.resolve())
    args.text_model = str(args.text_model.resolve())
    args.fairseq_root = str(args.fairseq_root.resolve())
    args.deepecg_checkpoint = str(args.deepecg_checkpoint.resolve())

    args.resume = str(args.resume.resolve()) if args.resume is not None else None
    args.ecg_model = 'ecg_clip_deepecg_init' if args.backbone_init == 'deepecg_ssl' else 'ecg_clip_transformer'
    args.use_fp16 = args.amp_dtype == 'bf16'
    args.txt_transform = True
    args.verbose_epoch = -1
    args.saveckp_freq = 5
    return args

def _group_l2_norms(model: torch.nn.Module, *, gradients: bool) -> dict[str, torch.Tensor]:
    """Compute aggregate L2 norms without concatenating model parameters."""
    grouped: dict[str, list[torch.Tensor]] = {'ecg_encoder': [], 'text_encoder': [], 'projection_heads': []}
    module = model.module if hasattr(model, 'module') else model
    for (name, parameter) in module.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.grad if gradients else parameter
        if value is None:
            continue
        if name.startswith('ecg_encoder.'):
            group = 'ecg_encoder'
        elif name.startswith('lm_model.'):
            group = 'text_encoder'
        else:
            group = 'projection_heads'
        grouped[group].append(value.detach().float().norm(2).square())
    result: dict[str, torch.Tensor] = {}
    for (group, squared_norms) in grouped.items():
        if squared_norms:
            result[group] = torch.stack(squared_norms).sum().sqrt()
    return result

def _similarity_monitor(ecg_1: torch.Tensor, ecg_2: torch.Tensor, proj_ecg: torch.Tensor, proj_text: torch.Tensor, report_template_ids: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
    proj_ecg = torch.nn.functional.normalize(proj_ecg.detach().float(), dim=-1)
    proj_text = torch.nn.functional.normalize(proj_text.detach().float(), dim=-1)
    ecg_1 = torch.nn.functional.normalize(ecg_1.detach().float(), dim=-1)
    ecg_2 = torch.nn.functional.normalize(ecg_2.detach().float(), dim=-1)
    cma = proj_ecg @ proj_text.T
    batch = cma.shape[0]
    offdiag = ~torch.eye(batch, dtype=torch.bool, device=cma.device)
    result = {'uma_positive_cosine_mean': (ecg_1 * ecg_2).sum(dim=-1).mean()}
    if report_template_ids is None:
        result.update({'cma_positive_cosine_mean': cma.diagonal().mean(), 'cma_offdiag_cosine_mean': cma[offdiag].mean() if batch > 1 else cma.new_zeros(()), 'cma_positive_offdiag_gap': cma.diagonal().mean() - cma[offdiag].mean() if batch > 1 else cma.new_zeros(())})
    else:
        if report_template_ids.shape != (batch,):
            raise ValueError('Report-template IDs do not match the global CMA batch')
        positive_mask = report_template_ids[:, None].eq(report_template_ids[None, :])
        negative_mask = ~positive_mask
        positive_mean = cma[positive_mask].mean()
        negative_mean = cma[negative_mask].mean() if bool(negative_mask.any()) else cma.new_zeros(())
        result.update({'cma_multi_positive_cosine_mean': positive_mean, 'cma_negative_cosine_mean': negative_mean, 'cma_positive_negative_gap': positive_mean - negative_mean, 'cma_diagonal_cosine_mean': cma.diagonal().mean()})
    return result

def _make_monitored_train_one_epoch(original):
    """Preserve the public loop and add aggregate TensorBoard diagnostics."""

    def train_one_epoch(model, data_loader, optimizer, lr_schedule, epoch, fp16_scaler, logger, args, writer):
        metric_logger = original.MetricLogger(delimiter='  ')
        header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
        if not hasattr(data_loader.sampler, 'set_epoch'):
            raise TypeError('Distributed training sampler must provide set_epoch(epoch)')
        data_loader.sampler.set_epoch(epoch)
        for (it, subjects_batch) in enumerate(metric_logger.log_every(data_loader, 10, header, logger)):
            ecg = subjects_batch['ecg_data'].cuda(non_blocking=True)
            report = subjects_batch['report']
            global_steps = epoch * len(data_loader) + it
            monitor_now = global_steps % args.monitor_interval == 0
            freeze_ecg = global_steps < args.freeze_ecg_backbone_updates
            model.module.set_ecg_backbone_frozen(freeze_ecg)
            with torch.cuda.amp.autocast(enabled=args.amp_dtype == 'bf16', dtype=torch.bfloat16):
                report_tokens = model.module._tokenize(report)
                input_ids = report_tokens.input_ids.cuda(non_blocking=True)
                attention_mask = report_tokens.attention_mask.cuda(non_blocking=True)
                output_dict = model(ecg, input_ids, attention_mask)
                (ecg_emb_1, ecg_emb_2) = output_dict['ecg_emb']
                agg_ecg_emb_1 = original.distributed_concat(ecg_emb_1)
                agg_ecg_emb_2 = original.distributed_concat(ecg_emb_2)
                agg_proj_ecg_emb = original.distributed_concat(output_dict['proj_ecg_emb'])
                agg_proj_text_emb = original.distributed_concat(output_dict['proj_text_emb'])
                agg_report_template_ids = None
                cma_metrics: dict[str, torch.Tensor] = {}
                if args.cma_loss == 'multi_positive':
                    local_template_ids = subjects_batch['report_template_id'].cuda(non_blocking=True)
                    agg_report_template_ids = distributed_concat_ids(local_template_ids)
                    (cma_loss, acc1, acc5, cma_metrics) = multi_positive_clip_loss(agg_proj_ecg_emb, agg_proj_text_emb, agg_report_template_ids, temperature=args.contrastive_temperature)
                else:
                    (cma_loss, acc1, acc5) = original.clip_loss(agg_proj_ecg_emb, agg_proj_text_emb)
                (uma_loss, _, _) = original.clip_loss(agg_ecg_emb_1, agg_ecg_emb_2)
                clip_loss = cma_loss + args.uma_weight * uma_loss

                loss = clip_loss
                res_dict = {'loss_cma': cma_loss, 'loss_uma': uma_loss, 'loss_clip': clip_loss, 'loss': loss, 'acc1': acc1, 'acc5': acc5}
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.unscale_(optimizer)
            monitor: dict[str, torch.Tensor] = {}
            if monitor_now:
                monitor['ecg_backbone_frozen'] = ecg.new_tensor(float(freeze_ecg))

                monitor.update(model.module.last_monitor)
                monitor.update(cma_metrics)
                monitor.update(_similarity_monitor(agg_ecg_emb_1, agg_ecg_emb_2, agg_proj_ecg_emb, agg_proj_text_emb, agg_report_template_ids))
                if args.log_gradient_norms:
                    for (name, value) in _group_l2_norms(model, gradients=True).items():
                        monitor[f'grad_norm_{name}'] = value
                    for (name, value) in _group_l2_norms(model, gradients=False).items():
                        monitor[f'param_norm_{name}'] = value
            clip_threshold = args.max_grad_norm if args.max_grad_norm > 0 else float('inf')
            pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_threshold, error_if_nonfinite=True)
            if monitor_now:
                monitor['grad_norm_global_pre_clip'] = pre_clip_grad_norm.detach().float()
                post_clip_groups = _group_l2_norms(model, gradients=True)
                if post_clip_groups:
                    monitor['grad_norm_global_post_clip'] = torch.stack([value.square() for value in post_clip_groups.values()]).sum().sqrt()
                monitor = original.reduce_dict(monitor)
            if fp16_scaler is None:
                optimizer.step()
            else:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            if original.is_main_process():
                for (key, value) in res_dict.items():
                    writer.add_scalar(f'train/{key}', value.item() if isinstance(value, torch.Tensor) else value, global_steps)
                if monitor_now:
                    writer.add_scalar('optimization/lr', optimizer.param_groups[0]['lr'], global_steps)
                    for group in optimizer.param_groups:
                        group_name = group.get('group_name')
                        if group_name:
                            writer.add_scalar(f'optimization/lr_{group_name}', group['lr'], global_steps)
                    writer.add_scalar('optimization/weight_decay', optimizer.param_groups[0]['weight_decay'], global_steps)
                    for (key, value) in monitor.items():
                        writer.add_scalar(f'intermediate/{key}', value.item() if isinstance(value, torch.Tensor) else value, global_steps)
            lr_schedule.step()
            metric_logger.update(loss_train=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])
            metric_logger.update(wd=optimizer.param_groups[0]['weight_decay'])
        metric_logger.synchronize_between_processes()
        return {key: meter.global_avg for (key, meter) in metric_logger.meters.items()}
    return train_one_epoch

def _to_float(value) -> float:
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)

def _update_early_stopping(state: dict[str, object], *, valid_loss: float, epoch: int, patience: int, min_epochs: int, min_delta: float) -> bool:
    """Update serializable early-stop state and return a deterministic stop flag."""
    state.setdefault('early_stopping_best_loss', np.inf)
    state.setdefault('early_stopping_bad_epochs', 0)
    meaningful_improvement = valid_loss < float(state['early_stopping_best_loss']) - float(min_delta)
    if meaningful_improvement:
        state['early_stopping_best_loss'] = float(valid_loss)
        state['early_stopping_bad_epochs'] = 0
    elif patience > 0 and epoch + 1 >= min_epochs:
        state['early_stopping_bad_epochs'] = int(state['early_stopping_bad_epochs']) + 1
    else:
        state['early_stopping_bad_epochs'] = 0
    return bool(patience > 0 and epoch + 1 >= min_epochs and (int(state['early_stopping_bad_epochs']) >= patience))

def _resolve_scheduler_t0(args: argparse.Namespace, world_size: int) -> int:
    current_global_batch = int(args.train_batch_size_per_gpu) * int(world_size)
    if current_global_batch <= 0:
        raise ValueError('Current global batch must be positive')
    if args.scheduler_t0 is not None:
        return int(args.scheduler_t0)
    return max(1, round(args.scheduler_reference_t0 * args.scheduler_reference_global_batch / current_global_batch))

def _build_lr_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> torch.optim.lr_scheduler.LRScheduler:
    if args.lr_scheduler_type == 'single_cosine':
        total_updates = int(args.total_updates_resolved)
        warmup_updates = int(args.warmup_updates)
        if total_updates <= warmup_updates:
            raise ValueError('single_cosine requires total updates greater than warmup updates')
        min_ratio = float(args.scheduler_min_lr_ratio)

        def multiplier(step: int) -> float:
            if warmup_updates > 0 and step < warmup_updates:
                return max(1.0 / warmup_updates, (step + 1.0) / warmup_updates)
            progress = (step - warmup_updates) / float(total_updates - warmup_updates)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_t0_resolved, T_mult=1, eta_min=1e-08)
    if args.warmup_updates == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / float(args.warmup_updates), end_factor=1.0, total_iters=args.warmup_updates)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_updates])

def _valid_one_epoch_multi_positive(original, model, data_loader, epoch, writer, logger, args):
    """Validation equivalent of the historical loop with report-aware CMA."""
    metric_logger = original.MetricLogger(delimiter='  ')
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    totals: dict[str, float] = {'loss_cma': 0.0, 'loss_uma': 0.0, 'loss_clip': 0.0, 'loss': 0.0, 'acc1': 0.0, 'acc5': 0.0, 'report_positive_count_mean': 0.0, 'report_global_batch_size': 0.0, 'report_positive_count_max': 0.0, 'report_duplicate_anchor_fraction': 0.0, 'report_unique_template_fraction': 0.0, 'loss_cma_ecg_to_text': 0.0, 'loss_cma_text_to_ecg': 0.0}
    count = 0
    for subjects_batch in metric_logger.log_every(data_loader, 10, header, logger):
        ecg = subjects_batch['ecg_data'].cuda(non_blocking=True)
        report = subjects_batch['report']
        template_ids = subjects_batch['report_template_id'].cuda(non_blocking=True)
        report_tokens = model.module._tokenize(report)
        input_ids = report_tokens.input_ids.cuda(non_blocking=True)
        attention_mask = report_tokens.attention_mask.cuda(non_blocking=True)
        output_dict = model(ecg, input_ids, attention_mask)
        (ecg_emb_1, ecg_emb_2) = output_dict['ecg_emb']
        agg_ecg_emb_1 = original.distributed_concat(ecg_emb_1)
        agg_ecg_emb_2 = original.distributed_concat(ecg_emb_2)
        agg_proj_ecg = original.distributed_concat(output_dict['proj_ecg_emb'])
        agg_proj_text = original.distributed_concat(output_dict['proj_text_emb'])
        agg_template_ids = distributed_concat_ids(template_ids)
        (cma_loss, acc1, acc5, cma_metrics) = multi_positive_clip_loss(agg_proj_ecg, agg_proj_text, agg_template_ids, temperature=args.contrastive_temperature)
        (uma_loss, _, _) = original.clip_loss(agg_ecg_emb_1, agg_ecg_emb_2)
        clip_loss = cma_loss + args.uma_weight * uma_loss

        loss = clip_loss
        values = {'loss_cma': cma_loss, 'loss_uma': uma_loss, 'loss_clip': clip_loss, 'loss': loss, 'acc1': acc1, 'acc5': acc5, **cma_metrics}
        for key in totals:
            value = values[key]
            totals[key] += float(value.item() if isinstance(value, torch.Tensor) else value)
        count += 1
    if count == 0:
        raise RuntimeError('Validation loader produced no batches')
    averaged = {key: value / count for (key, value) in totals.items()}
    reduced = original.reduce_dict(averaged)
    if original.is_main_process():
        for key in ('loss_cma', 'loss_uma', 'loss_clip', 'loss', 'acc1', 'acc5'):
            writer.add_scalar(f'valid/{key}', _to_float(reduced[key]), epoch)
        for key in totals:
            if key not in {'loss_cma', 'loss_uma', 'loss_clip', 'loss', 'acc1', 'acc5'}:
                writer.add_scalar(f'valid_report/{key}', _to_float(reduced[key]), epoch)
    return reduced

def _train_valid_one_epoch(original, model, train_loader, valid_loader, epoch, best_valid_stats, optimizer, lr_scheduler, fp16_scaler, logger, args, writer):
    start_time = time.time()
    model.train()
    train_stats = _make_monitored_train_one_epoch(original)(model, train_loader, optimizer, lr_scheduler, epoch, fp16_scaler, logger, args, writer)
    model.eval()
    with torch.no_grad():
        if args.cma_loss == 'multi_positive':
            valid_stats = _valid_one_epoch_multi_positive(original, model, valid_loader, epoch, writer, logger, args)
        else:
            valid_stats = original.valid_one_epoch(model=model, data_loader=valid_loader, epoch=epoch, writer=writer, logger=logger, args=args)
    valid_loss = _to_float(valid_stats.get('loss_clip', valid_stats['loss']))
    valid_acc = _to_float(valid_stats['acc1'])
    previous = dict(best_valid_stats)
    is_best_loss = valid_loss < float(previous['valid_loss'])
    is_best_acc = valid_acc > float(previous['valid_acc'])
    if is_best_loss:
        best_valid_stats['valid_loss'] = valid_loss
        best_valid_stats['best_loss_epoch'] = epoch
    if is_best_acc:
        best_valid_stats['valid_acc'] = valid_acc
        best_valid_stats['best_acc_epoch'] = epoch
    should_stop = _update_early_stopping(best_valid_stats, valid_loss=valid_loss, epoch=epoch, patience=args.early_stopping_patience, min_epochs=args.early_stopping_min_epochs, min_delta=args.early_stopping_min_delta)
    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch + 1, 'best_valid_stats': dict(best_valid_stats), 'args': args}
    if fp16_scaler is not None:
        save_dict['fp16_scaler'] = fp16_scaler.state_dict()
    checkpoint_path = os.path.join(args.experiment_path, f'{args.exp_name}-checkpoint.pth')
    original.save_on_master(save_dict, checkpoint_path)
    if args.save_encoder_every_epoch:
        module = model.module if hasattr(model, 'module') else model
        encoder_state = {f'ecg_encoder.{key}': value for (key, value) in module.ecg_encoder.state_dict().items()}
        epoch_directory = Path(args.experiment_path) / 'epoch_encoders'
        epoch_path = epoch_directory / f'{args.exp_name}-ecg-encoder-epoch{epoch:02d}.pt'
        if original.is_main_process():
            epoch_directory.mkdir(parents=True, exist_ok=True)
            if epoch_path.exists():
                raise FileExistsError(f'Refusing to overwrite per-epoch encoder checkpoint: {epoch_path}')
        original.save_on_master({'model': encoder_state, 'epoch': epoch + 1, 'best_valid_stats': dict(best_valid_stats), 'args': args, 'checkpoint_type': 'ecg_encoder_only_for_linear_probe'}, str(epoch_path))
    if is_best_loss:
        old_epoch = int(previous['best_loss_epoch'])
        old_loss = float(previous['valid_loss'])
        if old_epoch >= 0:
            old_path = os.path.join(args.experiment_path, f'{args.exp_name}-model-epoch{old_epoch}-{old_loss:.2f}_loss.pt')
            if original.is_main_process() and os.path.isfile(old_path):
                os.remove(old_path)
        new_path = os.path.join(args.experiment_path, f'{args.exp_name}-model-epoch{epoch}-{valid_loss:.2f}_loss.pt')
        original.save_on_master(save_dict, new_path)
    if is_best_acc:
        old_epoch = int(previous['best_acc_epoch'])
        old_acc = float(previous['valid_acc'])
        if old_epoch >= 0:
            old_path = os.path.join(args.experiment_path, f'{args.exp_name}-model-epoch{old_epoch}-{old_acc:.2f}_acc.pt')
            if original.is_main_process() and os.path.isfile(old_path):
                os.remove(old_path)
        new_path = os.path.join(args.experiment_path, f'{args.exp_name}-model-epoch{epoch}-{valid_acc:.2f}_acc.pt')
        original.save_on_master(save_dict, new_path)
    log_stats = {**{f'train_{key}': value for (key, value) in train_stats.items()}, **{f'valid_{key}': value for (key, value) in valid_stats.items()}, 'epoch': epoch, 'checkpoint_selection_clip_loss': valid_loss, 'scheduler_t0': args.scheduler_t0_resolved, 'early_stopping_bad_epochs': int(best_valid_stats['early_stopping_bad_epochs']), 'early_stopping_best_loss': float(best_valid_stats['early_stopping_best_loss']), 'early_stopping_should_stop': int(should_stop)}
    if original.is_main_process():
        original.log_loss_to_file(log_stats, logger=logger)
        writer.add_scalar('early_stopping/bad_epochs', int(best_valid_stats['early_stopping_bad_epochs']), epoch)
        writer.add_scalar('early_stopping/best_loss', float(best_valid_stats['early_stopping_best_loss']), epoch)
        writer.add_scalar('early_stopping/should_stop', int(should_stop), epoch)
        writer.flush()
    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    original.print_log(f'Training time {elapsed}', logger=logger)
    return (train_stats, valid_stats, should_stop)

def _train_transformer(original, args: argparse.Namespace) -> None:
    original.init_distributed_mode(args)
    original.fix_random_seeds(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.amp_dtype == 'bf16' and (not torch.cuda.is_bf16_supported()):
        raise RuntimeError('The selected GPU does not support CUDA BF16 autocast')
    current_global_batch = int(args.train_batch_size_per_gpu) * int(args.world_size)
    args.scheduler_t0_resolved = _resolve_scheduler_t0(args, args.world_size)
    args.train_global_batch = current_global_batch
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = original.get_root_logger(log_file=log_file, name=args.log_name)
    original.print_log('==================== Parameters ====================', logger=logger)
    original.log_args_to_file(args, 'args', logger=logger)
    original.print_log('==================== preparing data ... ====================', logger=logger)
    data_root = Path(args.data_root)
    need_report_template_ids = args.cma_loss == 'multi_positive' or args.report_sampling_alpha > 0.0

    (train_ds, valid_ds) = ReadOnlyECGPretrainDataset.create_datasets(data_root / args.h5_file_path, data_root / args.label_file_path, data_root / args.info_file_path, data_root / args.train_indices_file_path, data_root / args.valid_indices_file_path, args.num_leads, include_report_template_ids=need_report_template_ids)
    original.print_log(f'train dataset load complete: {len(train_ds)} image-text pairs', logger=logger)
    original.print_log(f'valid dataset load complete: {len(valid_ds)} image-text pairs', logger=logger)
    original.print_log(f'ECG-text objective: cma_loss={args.cma_loss}; temperature={args.contrastive_temperature}; report_sampling_alpha={args.report_sampling_alpha}; uma_weight={args.uma_weight}', logger=logger)

    if args.report_sampling_alpha > 0.0:
        sampling_weights = train_ds.get_report_sampling_weights(args.report_sampling_alpha)
        train_sampler = DistributedTemplateWeightedSampler(train_ds, sampling_weights, num_replicas=args.world_size, rank=args.rank, seed=args.seed, drop_last=False)
        sampling_summary = train_ds.report_sampling_summary(args.report_sampling_alpha)
        sampling_summary['expected_same_template_others_global_batch'] = sampling_summary['same_template_collision_probability'] * (current_global_batch - 1)
        args.report_sampling_summary = sampling_summary
        original.print_log(f"Report-template weighted sampling: alpha={args.report_sampling_alpha}; rows_per_global_epoch={train_sampler.total_size}; unique_templates={sampling_summary['unique_template_count']}; top_template_probability={sampling_summary['top_template_probability']:.6f}; expected_same_template_others_global_batch={sampling_summary['expected_same_template_others_global_batch']:.3f}", logger=logger)
    else:
        train_sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        args.report_sampling_summary = {'alpha': 0.0, 'sampler': 'uniform_distributed_rows'}
    valid_sampler = torch.utils.data.DistributedSampler(valid_ds, shuffle=False)
    loader_worker_options = {}
    if args.num_workers > 0:
        loader_worker_options = {'persistent_workers': True, 'prefetch_factor': 4}
    train_loader = torch.utils.data.DataLoader(train_ds, sampler=train_sampler, batch_size=args.train_batch_size_per_gpu, num_workers=args.num_workers, pin_memory=True, drop_last=True, **loader_worker_options)
    valid_loader = torch.utils.data.DataLoader(valid_ds, sampler=valid_sampler, batch_size=args.valid_batch_size_per_gpu, num_workers=args.num_workers, pin_memory=True, drop_last=False, **loader_worker_options)
    args.total_updates_resolved = int(args.epochs) * len(train_loader)
    original.print_log('==================== building network ... ====================', logger=logger)
    model = _build_model(args).cuda()
    if hasattr(model, 'initial_backbone_load_report'):
        (missing, unexpected) = model.initial_backbone_load_report
        original.print_log(f"DeepECG initialization: missing_keys={missing}; unexpected_keys={unexpected}; normalization={args.deepecg_normalization_mode}; input_scale={args.deepecg_input_scale}; gradient_checkpointing={getattr(model.ecg_encoder, 'gradient_checkpointing', False)}; checkpointed_encoder_layers={getattr(model.ecg_encoder, 'checkpointed_encoder_layers', 0)}", logger=logger)
    if not args.freeze_text_encoder:
        if args.text_encoder_freeze_layers is not None:
            for layer_idx in range(int(args.text_encoder_freeze_layers)):
                for parameter in model.lm_model.encoder.layer[layer_idx].parameters():
                    parameter.requires_grad = False
    else:
        for parameter in model.lm_model.parameters():
            parameter.requires_grad = False
    if original.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.gpu])
    fp16_scaler = None
    trainable_named = [(name, parameter) for (name, parameter) in model.named_parameters() if parameter.requires_grad]
    if args.ecg_backbone_lr is None and args.text_encoder_lr is None:
        parameter_groups = [{'params': [parameter for (_, parameter) in trainable_named], 'lr': args.lr, 'group_name': 'all'}]
    elif args.text_encoder_lr is None:
        ecg_parameters = [parameter for (name, parameter) in trainable_named if name.startswith('module.ecg_encoder.')]
        other_parameters = [parameter for (name, parameter) in trainable_named if not name.startswith('module.ecg_encoder.')]
        if not ecg_parameters or not other_parameters:
            raise RuntimeError('Differential LR requires both ecg_encoder and non-ECG trainable parameters')
        parameter_groups = [{'params': other_parameters, 'lr': args.lr, 'group_name': 'projection_text'}, {'params': ecg_parameters, 'lr': args.ecg_backbone_lr, 'group_name': 'ecg_backbone'}]
    else:
        ecg_parameters = [parameter for (name, parameter) in trainable_named if name.startswith('module.ecg_encoder.')]
        text_parameters = [parameter for (name, parameter) in trainable_named if name.startswith('module.lm_model.')]
        head_parameters = [parameter for (name, parameter) in trainable_named if not name.startswith('module.ecg_encoder.') and (not name.startswith('module.lm_model.'))]
        if not ecg_parameters or not text_parameters or (not head_parameters):
            raise RuntimeError('Three-way differential LR requires non-empty ECG, text and head groups')
        parameter_groups = [{'params': head_parameters, 'lr': args.lr, 'group_name': 'projection_heads'}, {'params': text_parameters, 'lr': args.text_encoder_lr, 'group_name': 'text_encoder'}, {'params': ecg_parameters, 'lr': args.ecg_backbone_lr or args.lr, 'group_name': 'ecg_backbone'}]
    optimizer = torch.optim.AdamW(parameter_groups, betas=(args.momentum, args.adam_beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
    lr_scheduler = _build_lr_scheduler(optimizer, args)
    if args.lr_scheduler_type == 'single_cosine':
        scheduler_description = f'Scheduler: synchronized linear warmup then single cosine; warmup_updates={args.warmup_updates}; total_updates={args.total_updates_resolved}; final_lr_ratio={args.scheduler_min_lr_ratio}'
    else:
        scheduler_description = f'Scheduler: linear warmup then cosine restarts aligned by samples; warmup_updates={args.warmup_updates}; reference T_0={args.scheduler_reference_t0} at global batch {args.scheduler_reference_global_batch}; resolved T_0={args.scheduler_t0_resolved} at global batch {current_global_batch}'
    original.print_log(scheduler_description, logger=logger)
    original.print_log(f'Global gradient clipping max_norm={args.max_grad_norm} (0 means monitor only; non-finite gradients are fatal)', logger=logger)
    original.print_log(f'AdamW aligned to official fairseq-signals: betas=({args.momentum}, {args.adam_beta2}), eps={args.adam_eps}, weight_decay={args.weight_decay}; projection/head lr={args.lr}; text encoder lr={args.text_encoder_lr or args.lr}; ECG backbone lr={args.ecg_backbone_lr or args.lr}; freeze ECG backbone updates={args.freeze_ecg_backbone_updates}', logger=logger)
    best_valid_stats = {'best_loss_epoch': -1, 'valid_loss': np.inf, 'best_acc_epoch': -1, 'valid_acc': -np.inf, 'early_stopping_best_loss': np.inf, 'early_stopping_bad_epochs': 0}
    to_restore = {'epoch': 0, 'best_valid_stats': best_valid_stats}
    if args.resume:
        checkpoint_objects = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        if fp16_scaler is not None:
            checkpoint_objects['fp16_scaler'] = fp16_scaler
        original.restart_from_checkpoint(args.resume, run_variables=to_restore, **checkpoint_objects)
    start_epoch = int(to_restore['epoch'])
    best_valid_stats = dict(to_restore['best_valid_stats'])
    best_valid_stats.setdefault('early_stopping_best_loss', np.inf)
    best_valid_stats.setdefault('early_stopping_bad_epochs', 0)
    writer = original.create_tfboard_on_master(args.tfboard_path)
    original.print_log(f'Starting training at epoch {start_epoch}', logger=logger)
    original.print_log(f'Early stopping: enabled={args.early_stopping_patience > 0}; monitor=validation CMA + weighted UMA; min_epochs={args.early_stopping_min_epochs}; patience={args.early_stopping_patience}; min_delta={args.early_stopping_min_delta}', logger=logger)
    try:
        for epoch in range(start_epoch, args.epochs):
            (_, _, should_stop) = _train_valid_one_epoch(original, model, train_loader, valid_loader, epoch, best_valid_stats, optimizer, lr_scheduler, fp16_scaler, logger, args, writer)
            stop_tensor = torch.tensor([int(should_stop)], dtype=torch.int32, device=torch.device('cuda'))
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(stop_tensor, src=0)
            if bool(stop_tensor.item()):
                original.print_log(f"Early stopping triggered after epoch {epoch}; best monitored loss={best_valid_stats['early_stopping_best_loss']:.6g}; bad_epochs={best_valid_stats['early_stopping_bad_epochs']}", logger=logger)
                break
    finally:
        if original.is_main_process() and writer is not None:
            writer.flush()
            writer.close()

def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        _dry_run(args)
        return
    _validate_args(args)
    prepare_public_source_imports()
    import main as original

    def destroy_process_group_at_exit() -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    atexit.register(destroy_process_group_at_exit)
    _train_transformer(original, _formal_args(args))
if __name__ == '__main__':
    main()
