from __future__ import annotations
# Most functions are reused from DINO

import os
import sys
import time
import math
import random
import argparse
import datetime
import subprocess
from tqdm import tqdm
import warnings
from collections import defaultdict, deque
from .logger import *
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from sklearn.metrics import (
    recall_score, 
    precision_recall_curve, 
    auc, 
    confusion_matrix, 
    f1_score, roc_curve, 
    roc_auc_score, 
    precision_score,
    matthews_corrcoef, 
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from collections.abc import Iterable


def load_state_with_same_shape(model, weights, model_type, load_module_name='image_encoder'):
    model_state = model.state_dict()
    # remove ddp
    if list(weights.keys())[0].startswith('module.'):
        weights = {k.partition('module.')[-1]: weights[k] for k in weights.keys()}

    if load_module_name == 'image_encoder':
        # load vit params
        if model_type == 'student':
            load_weights = {k.replace('image_encoder_q_student.', ''): weights[k] for k in weights if
                                k.startswith('image_encoder_q_student')}
        elif model_type == 'teacher':
            load_weights = {k.replace('image_encoder_q_teacher.', ''): weights[k] for k in weights if
                                k.startswith('image_encoder_q_teacher')}
        else:
            raise NotImplementedError(f'model_type {model_type} not implemented!')
    elif load_module_name == 'fusion_model':
        # load fusion model params
        if model_type == 'student':
            load_weights = {k.replace('fusion_model_student.', ''): weights[k] for k in weights if
                                k.startswith('fusion_model_student')}
        elif model_type == 'teacher':
            load_weights = {k.replace('fusion_model_teacher.', ''): weights[k] for k in weights if
                                k.startswith('fusion_model_teacher')}
        else:
            raise NotImplementedError(f'model_type {model_type} not implemented!')

    elif load_module_name == 'text_encoder_model':
        if model_type == '':
            load_weights = {k.replace('text_encoder_q.model.', ''): weights[k] for k in weights if
                            k.startswith('text_encoder_q.model')}
        else:
            load_weights = {k.replace(f'text_encoder_q_{model_type}.model.', ''): weights[k] for k in weights if
                            k.startswith(f'text_encoder_q_{model_type}.model')}

    elif load_module_name == 'text_encoder_global_embed':
        if model_type == '':
            load_weights = {k.replace('text_encoder_q.global_embed.', ''): weights[k] for k in weights if
                            k.startswith('text_encoder_q.global_embed')}
        else:
            load_weights = {k.replace(f'text_encoder_q_{model_type}.global_embed.', ''): weights[k] for k in weights if
                            k.startswith(f'text_encoder_q_{model_type}.global_embed')}
    else:
        raise NotImplementedError(f"load_module_name: {load_module_name} is not implemented! Should in [image_encoder, text_encoder, fusion_model]")
    load_weights = {**load_weights}
    filtered_weights = {
        k: v for k, v in load_weights.items() if k in model_state and v.size() == model_state[k].size()
    }

    print(f"Loading {load_module_name} weights:" + ', '.join(filtered_weights.keys()))

    return filtered_weights

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# EMA 算法更新teacher的参数（follow student）
class EMA:
    def __init__(self):
        super().__init__()

    def update_average(self, old, new, beta):
        if old is None:
            return new
        return old * beta + (1 - beta) * new


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded {} from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def parse_tuple(value):
    return tuple(value.split(','))


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_log(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB), logger=logger)
                else:
                    print_log(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)), logger=logger)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_log('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)), logger=logger)

    def log_every_list(self, iterable_list, print_freq, header=None):
        i = 0
        iterable2, iterable1 = iterable_list
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable1)))) + 'd'
        # space_fmt = ':d'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in zip(iterable2, iterable1):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable1) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable1) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable1), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable1), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable1)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def remove_on_master(filename_to_remove):
    if is_main_process():
        os.remove(filename_to_remove)


def create_tfboard_on_master(tfboard_to_save):
    if is_main_process():
        writer = SummaryWriter(tfboard_to_save)

        return writer


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        # args.dist_url = "tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
        # print(args.rank, args.world_size, args.gpu, args.dist_url)
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def get_params(model):
    params_to_update = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        else:
            params_to_update.append(param)
            
    return {'params': params_to_update}


def get_params_with_name(model):
    params_to_update = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        else:
            params_to_update[name] = param
    return params_to_update


def get_params_groups(model, base_lr, finetune_lr):
    vit_pretrained_params = [p for name, p in model.image_encoder_q_student.named_parameters() if
                             name not in model.vit_missing_keys]
    vit_new_params = [p for name, p in model.image_encoder_q_student.named_parameters() if
                      name in model.vit_missing_keys and '3D' in name]

    new_params = list(
        set(vit_new_params).union(set(model.fusion_model_student.parameters())))  # patch pe shared weights
    pretrained_params = list(set(vit_pretrained_params).union(set(model.text_encoder_q.parameters())))

    new_params_group = {'params': new_params, 'lr': base_lr}
    pretrained_params_group = {'params': pretrained_params, 'lr': finetune_lr}

    # print(fr'=========== new params: {new_params}')

    return [new_params_group, pretrained_params_group]


def get_params_group_llrd(model, base_lr=5e-5, layer_lr_decay=0.80, vit_depth=12, bert_depth=6, fusion_depth=2):
    clip_depth = vit_depth + fusion_depth
    llrd_info_dict = {}
    update_named_params = get_params_with_name(model)
    for full_name, params, in update_named_params.items():
        # remove module in ddp
        if full_name.startswith('module.'):
            name = full_name.partition('module.')[-1]
        else:
            name = full_name

        # 3 modules for CLIP training
        module_list = ["image_encoder_q_student", "text_encoder_q", "fusion_model_student"]

        if name.startswith("image_encoder_q_student"):
            name = name.partition("image_encoder_q_student.")[-1]
            # if name in ("cls_token3D", "pos_embed3D", "patch_embed3D.proj.weight", "patch_embed3D.proj.bias"):
            if name.startswith("cls") or name.startswith("pos") or name.startswith("patch"):
                layer_id = 0
            elif name.startswith("blocks"):
                layer_id = int(name.split('.')[1]) + 1
            elif name.startswith("norm") or name.startswith("global_embedding"):
                layer_id = vit_depth
            else:
                print(full_name)

        elif name.startswith("text_encoder_q"):
            name = name.partition("text_encoder_q.")[-1]
            if name.startswith("model"):
                name = name.partition("model.")[-1]
                # text_encoder_q.model.embeddings.word_embeddings.weight
                if name.startswith("embeddings"):
                    layer_id = 0
                # text_encoder_q.model.encoder.layer.0.attention.self.query.weight
                elif name.startswith("encoder"):
                    layer_id = int(name.split('.')[2]) + 1
                else:
                    print(full_name)
            elif name.startswith("global_embed"):
                layer_id = bert_depth

        elif name.startswith("fusion_model_student"):
            name = name.partition("fusion_model_student.")[-1]
            if name in ("cls_token4D", "modality_embed"):
                layer_id = 0 + vit_depth
            elif name.startswith("blocks"):
                layer_id = int(name.split('.')[1]) + 1 + vit_depth
            elif name.startswith("norm") or name.startswith("global_embedding"):
                layer_id = vit_depth + fusion_depth
            else:
                print(full_name)
        else:
            print(full_name)

        llrd_info_dict[full_name] = layer_id

    # grouped by layer_id
    llrd_info_grouped_dict = {}
    for key, value in llrd_info_dict.items():
        if value in llrd_info_grouped_dict:
            llrd_info_grouped_dict[value].append(key)
        else:
            llrd_info_grouped_dict[value] = [key]

    update_parameters_llrd = []
    named_parameters = {k: v for k, v in model.named_parameters()}
    for layer_id, params_names in llrd_info_grouped_dict.items():
        update_parameters_llrd += [{'params': [named_parameters[params_name] for params_name in params_names],
                        'lr': base_lr * layer_lr_decay ** (clip_depth - layer_id),
                                    'params_names': [params_name for params_name in params_names]}
                       ]

    return update_parameters_llrd

    # for item in parameters:
    #     print(item["params_name"], item["lr"])


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: tuple | float = (0.1, 0.01),
                 min_lr: tuple | float = (0.001, 0.0001),
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for (param_group, min_lr) in zip(self.optimizer.param_groups, self.min_lr):
            param_group['lr'] = min_lr
            self.base_lrs.append(min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for (base_lr, max_lr) in
                    zip(self.base_lrs, self.max_lr)]
        else:
            return [base_lr + (max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for (base_lr, max_lr) in zip(self.base_lrs, self.max_lr)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        # self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.max_lr = tuple(base_max_lr * (self.gamma ** self.cycle) for base_max_lr in self.base_max_lr)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            # print(lr)


# def get_metrics(logits_dict, labels_dict, task_info, detailed=False):
#     results = {}

#     for task_name, classes in task_info.items():
#         logits = logits_dict[task_name]
#         labels = labels_dict[task_name]

#         if isinstance(logits, torch.Tensor):
#             logits = logits.detach().cpu().numpy()
#         if isinstance(labels, torch.Tensor):
#             labels = labels.detach().cpu().numpy()

#         # 忽略标签为 -1 的样本
#         valid_mask = labels != -1
#         logits = logits[valid_mask]
#         labels = labels[valid_mask]

#         # 如果所有标签都为 -1，跳过当前任务的计算
#         if len(labels) == 0:
#             continue

#         metrics_dict = {}

#         if task_name == 'age':  # 回归任务
#             mse = mean_squared_error(labels, logits)
#             rmse = np.sqrt(mse)
#             mae = mean_absolute_error(labels, logits)
#             r2 = r2_score(labels, logits)

#             metrics_dict.update({
#                 "mean_squared_error": mse,
#                 "root_mean_squared_error": rmse,
#                 "mean_absolute_error": mae,
#                 "r2_score": r2,
#             })

#         else:  # 分类任务
#             # 稳定 softmax 的计算
#             logits = logits - np.max(logits, axis=1, keepdims=True)
#             # 计算概率
#             softmax_logits = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
#             pred_labels = np.argmax(logits, axis=1)

#             # 混淆矩阵
#             cm = confusion_matrix(labels, pred_labels, labels=np.arange(classes))

#             # 计算 ROC AUC（One-vs-Rest 方法）
#             if classes == 2:
#                 try:
#                     roc_auc = roc_auc_score(labels, softmax_logits[:, 1])  # 使用正类的概率
#                     metrics_dict["roc_auc"] = roc_auc
#                 except ValueError as e:
#                     print(f"Error in calculating ROC AUC for task {task_name}: {e}")
#                     continue
#             else:
#                 try:
#                     roc_auc = roc_auc_score(labels, softmax_logits, multi_class='ovr')
#                     metrics_dict["roc_auc"] = roc_auc
#                 except ValueError as e:
#                     print(f"Error in calculating ROC AUC for task {task_name}: {e}")
#                     continue

#             if classes == 2:
#                 # 二分类的特殊处理
#                 # print(labels)
#                 binarized_labels = np.zeros((labels.size, classes))
#                 binarized_labels[np.arange(labels.size), labels] = 1

#                 precisions, recalls, _ = precision_recall_curve(binarized_labels[:, 1], softmax_logits[:, 1])
#                 pr_auc = auc(recalls, precisions)

#                 specificity = np.sum(cm) - (np.sum(cm[:, 1]) + np.sum(cm[1, :]) - cm[1, 1])
#                 specificity = specificity / (specificity + np.sum(cm[:, 1]) - cm[1, 1]) if specificity + np.sum(cm[:, 1]) - cm[1, 1] != 0 else 0

#                 precision = precision_score(labels, pred_labels, zero_division=0)
#                 recall = recall_score(labels, pred_labels, zero_division=0)
#                 f1 = f1_score(labels, pred_labels, zero_division=0)

#                 # 更新结果字典
#                 metrics_dict.update({
#                     "pr_auc": pr_auc,
#                     "precision": precision,
#                     "recall": recall,
#                     "f1_score": f1,
#                     "specificity": specificity,
#                 })
#             else:
#                 # 多分类（包括三分类及以上）计算宏观平均指标
#                 binarized_labels = np.zeros((labels.size, classes))
#                 binarized_labels[np.arange(labels.size), labels] = 1

#                 pr_auc = {}
#                 for i in range(classes):
#                     precisions, recalls, _ = precision_recall_curve(binarized_labels[:, i], softmax_logits[:, i])
#                     pr_auc[i] = auc(recalls, precisions)
#                 # 宏观平均Precision-Recall AUC
#                 pr_auc_macro = np.mean(list(pr_auc.values()))

#                 # 计算每个类别的特异性
#                 specificity = []
#                 for i in range(classes):
#                     TN = np.sum(cm) - (np.sum(cm[:, i]) + np.sum(cm[i, :]) - cm[i, i])
#                     FP = np.sum(cm[:, i]) - cm[i, i]
#                     specificity_i = TN / (TN + FP) if TN + FP != 0 else 0
#                     specificity.append(specificity_i)
#                 # 计算平均特异性
#                 specificity_macro = np.mean(specificity)

#                 # 宏观平均指标
#                 precision_macro = precision_score(labels, pred_labels, average='macro', zero_division=0)
#                 recall_macro = recall_score(labels, pred_labels, average='macro', zero_division=0)
#                 f1_macro = f1_score(labels, pred_labels, average='macro', zero_division=0)

#                 # 更新结果字典
#                 metrics_dict.update({
#                     "pr_auc": pr_auc_macro,
#                     "precision": precision_macro,
#                     "recall": recall_macro,
#                     "f1_score": f1_macro,
#                     "specificity": specificity_macro,
#                 })

#         # 将当前任务的指标加入结果字典
#         results[task_name] = metrics_dict

#     return results



# def compute_mccs(gt, pred):
#     """
#     通过遍历 [0.0, 0.01, 0.02, ..., 1.0] 搜索最佳阈值。

#     参数:
#         - gt: 真实标签 (ground truth)，形状为 (n_samples,)。
#         - pred: 模型预测的概率 (logits)，形状为 (n_samples,)。
#     返回:
#         - best_threshold: 最佳的阈值（float）。
#         - best_mcc: 对应最佳阈值的 MCC 分数（float）。
#     """
#     best_threshold = 0.0
#     best_mcc = -1.0  # MCC 初始值设为 -1 确保有意义比较

#     # 遍历 0.0 到 1.0，每次增量 0.01
#     thresholds = np.arange(0.0, 1.01, 0.01)

#     for threshold in thresholds:
#         # 根据阈值生成二值化预测
#         pred_binary = (pred >= threshold).astype(int)
#         # 计算 MCC
#         mcc = matthews_corrcoef(gt, pred_binary)
#         # 更新最佳阈值和 MCC
#         if mcc > best_mcc:
#             best_mcc = mcc
#             best_threshold = threshold

#     return best_threshold, best_mcc


def search_best_threshold(gt, pred):
    """
    通过遍历 [0.0, 0.01, 0.02, ..., 1.0] 搜索最佳阈值。

    参数:
        - gt: 真实标签 (ground truth)，形状为 (n_samples,)。
        - pred: 模型预测的概率 (logits)，形状为 (n_samples,)。
    返回:
        - best_threshold: 最佳的阈值（float）。
        - best_f1_score: 对应最佳阈值的 F1 分数（float）。
    """
    best_threshold = 0.0
    best_f1 = 0.0

    # 遍历 0.0 到 1.0，每次增量 0.01
    thresholds = np.arange(0.0, 1.01, 0.01)

    for threshold in thresholds:
        # 根据阈值生成二值化预测
        pred_binary = (pred >= threshold).astype(int)
        # 计算 f1
        f1 = f1_score(gt, pred_binary)
        # 更新最佳阈值和 MCC
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def get_metrics(logits_dict, labels_dict, task_info, mode='valid', threshold=None):
    """
    计算多任务二分类的评估指标。
    
    参数:
        - logits_dict: 模型输出的 logits 字典，对于每个任务是一个 numpy 数组或 Tensor。
        - labels_dict: 模型对应的真实标签字典。
        - task_info: 每个任务的类别信息，字典格式。例如 {'Disease A': 2, 'Disease B': 2}。
        - mode: 当前阶段 ('train', 'valid', 'test')。
        - threshold: 仅在 'test' 阶段使用，传入的每个任务的阈值字典 {task_name: threshold}。
    返回:
        - results: 包含每个任务计算结果的字典。
        - thresholds: 在 'valid' 阶段返回的最佳阈值字典 (仅在 'valid' 阶段返回)。
    """
    results = {}
    thresholds = {}

    for task_name, classes in task_info.items():
        logits = logits_dict[task_name]
        labels = labels_dict[task_name]

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # 忽略标签为 -1 的样本
        valid_mask = labels != -1
        logits = logits[valid_mask]
        labels = labels[valid_mask]

        # 如果所有标签都为 -1，跳过当前任务的计算
        if len(labels) == 0:
            continue

        metrics_dict = {}

        if len(logits.shape) == 2:
            # 稳定 softmax 的计算
            logits = logits - np.max(logits, axis=1, keepdims=True)
            # 计算概率
            softmax_logits = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            predicted_probs = softmax_logits[:, 1]
        elif len(logits.shape) == 1:
            predicted_probs = logits

        if mode == 'train':  # 训练阶段，直接计算普通指标
            pred_binary = (predicted_probs >= 0.5).astype(int)  # 使用默认的阈值 0.5

            try:
                roc_auc = roc_auc_score(labels, predicted_probs)  # 使用正类的概率
                metrics_dict["roc_auc"] = roc_auc
            except ValueError as e:
                print(f"Error in calculating ROC AUC for task {task_name}: {e}")
                continue
            precision, recall, _ = precision_recall_curve(labels, predicted_probs)
            pr_auc = auc(recall, precision)

            # Precision, Recall, F1
            # precision_score_val = precision_score(labels, pred_binary)
            sensitivity = recall_score(labels, pred_binary)
            f1_score_val = f1_score(labels, pred_binary)
            mcc = matthews_corrcoef(labels, pred_binary)

            # 注意：如果要计算 specificity，需手动计算
            tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            metrics_dict.update({
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                # "precision": precision_score_val,
                "f1_score": f1_score_val,
                "mcc": mcc,
                "sensitivity": sensitivity,
                "specificity": specificity
            })

        elif mode == 'valid':  # 验证阶段，计算最佳阈值
            # 使用 search_best_threshold 计算最佳阈值
            # best_threshold, best_mcc = compute_mccs(labels, predicted_probs)
            best_threshold, best_f1 = search_best_threshold(labels, predicted_probs)
            thresholds[task_name] = best_threshold

            # 用最佳阈值计算指标
            pred_binary = (predicted_probs >= best_threshold).astype(int)
            mcc = matthews_corrcoef(labels, pred_binary)
            sensitivity = recall_score(labels, pred_binary)
            # 注意：如果要计算 specificity，需手动计算
            tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            # specificity = precision_score(labels, pred_binary, zero_division=0)
            # f1 = f1_score(labels, pred_binary)

            # ROC AUC 和 PR AUC
            roc_auc = roc_auc_score(labels, predicted_probs)
            precision, recall, _ = precision_recall_curve(labels, predicted_probs)
            pr_auc = auc(recall, precision)

            metrics_dict.update({
                "best_threshold": best_threshold,
                "mcc": mcc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1_score": best_f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            })

        elif mode == 'test':  # 测试阶段，直接使用传入的阈值
            if threshold is None or task_name not in threshold:
                raise ValueError(f"Thresholds must be provided for 'test' mode: missing `{task_name}`.")

            best_threshold = threshold[task_name]

            # 用传入的阈值计算指标
            pred_binary = (predicted_probs >= best_threshold).astype(int)

            sensitivity = recall_score(labels, pred_binary)
            # specificity = precision_score(labels, pred_binary, zero_division=0)
            # 注意：如果要计算 specificity，需手动计算
            tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = f1_score(labels, pred_binary)
            mcc = matthews_corrcoef(labels, pred_binary)

            # ROC AUC 和 PR AUC
            roc_auc = roc_auc_score(labels, predicted_probs)
            precision, recall, _ = precision_recall_curve(labels, predicted_probs)
            pr_auc = auc(recall, precision)

            metrics_dict.update({
                "threshold": best_threshold,
                "mcc": mcc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            })

        # 将当前任务的指标加入结果字典
        results[task_name] = metrics_dict

    if mode == 'valid':  # 如果是 valid 阶段，返回指标和最佳阈值
        return results, thresholds
    else:  # 其他阶段仅返回指标
        return results


# 检查对象是否为可迭代对象（排除字符串类型）
def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


# 扁平化列表
def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if is_iterable(item):
            flattened.extend(item)  # 如果是可迭代对象（如列表），使用 extend 扁平化
        else:
            flattened.append(item)  # 否则直接加入列表中
    return flattened


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待GPU上的所有任务完成
    return time.time()


def process_labels(batch_labels):
    # Check if batch_labels is a dictionary (multi-label situation)
    if isinstance(batch_labels, dict):
        for key in batch_labels:
            batch_labels[key] = batch_labels[key].cuda(non_blocking=True).squeeze()
    else:
        batch_labels = batch_labels.cuda(non_blocking=True).squeeze()
    
    return batch_labels


def custom_concat(left_labels, right_labels):
    # 检查是否是字典
    if isinstance(left_labels, dict) and isinstance(right_labels, dict):
        concatenated_labels = {}
        for key in left_labels:
            if key in right_labels:
                concatenated_labels[key] = torch.cat((left_labels[key], right_labels[key]), dim=0)
            else:
                raise KeyError(f"Key '{key}' not found in both dictionaries.")
        return concatenated_labels
    else:
        # 如果不是字典，直接进行连接
        return torch.cat((left_labels, right_labels), dim=0)
    
def _distributed_concat(tensor):
    gathered_tensors = torch.distributed.nn.all_gather(tensor)
    return torch.cat(gathered_tensors, dim=0)

def distributed_concat(inputs):
    if isinstance(inputs, dict):
        gathered_concat = {}
        for key in inputs:
            gathered_concat[key] = _distributed_concat(inputs[key])
        return gathered_concat
    else:
        return _distributed_concat(inputs)
    


def reduce_dict(input_dict):
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as input_dict, 
    after reduction.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        keys = []
        values = []
        # create separate lists of keys and values
        for k, v in sorted(input_dict.items()):
            keys.append(k)
            if not isinstance(v, torch.Tensor):
                v = torch.tensor([v]).cuda()
            values.append(v)

        values = torch.stack(values, dim=0)
        # All-reduce operation across all processes
        dist.all_reduce(values)
        # average by world_size
        values /= world_size
        # reconstruct the dictionary
        reduced_dict = {k: v.item() for k, v in zip(keys, values)}
    return reduced_dict
