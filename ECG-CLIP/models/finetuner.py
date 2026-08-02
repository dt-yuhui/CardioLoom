import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.svm import SVC
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from .resnet1d import ResNet18, ResNet34, ResNet50, ResNet101


class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.1) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if self.n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes)
            )
            
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class FineTuner(nn.Module):
    def __init__(self,
                 backbone,
                 in_features,
                 dropout,
                 task_info,
                 backbone_name='resnet',
                 hidden_dim=None,
                 finetune_type='linear_prob',
                 ):

        super().__init__()

        self.backbone = backbone
        self.in_features = in_features
        self.dropout = dropout
        self.task_info = task_info
        self.hidden_dim = hidden_dim
        self.finetune_type = finetune_type
        self.backbone_name = backbone_name
        self.task_nums = len(self.task_info)
        
        # if linear_prob, freeze backbone
        # else update all params
        if self.finetune_type == 'linear_prob' or self.finetune_type == 'svm':
            print("==================== freezing backbone ====================")
            for param in self.backbone.parameters():
                param.requires_grad = False

        if self.task_nums == 1:
            # task_info = {task_name : num_classes}
            (task_name, num_classes), = self.task_info.items()
            self.classifier = SSLEvaluator(
                    n_input=in_features, 
                    n_classes=num_classes, 
                    p=dropout, 
                    n_hidden=hidden_dim)
        else:
            task_nums = len(self.task_info)
            num_classes = 2
            self.classifier = SSLEvaluator(
                    n_input=in_features, 
                    n_classes=task_nums * num_classes, 
                    p=dropout, 
                    n_hidden=hidden_dim)
            

    def forward_features(self, batch):
        if self.backbone_name == 'resnet':
            # # backbone forward
            # if self.finetune_type == 'linear_prob' or self.finetune_type == 'svm':
            #     with torch.no_grad():
            #         feats = self.backbone(batch)
            # else:
            #     feats = self.backbone(batch)
            feats = self.backbone(batch)
            feats = self.backbone.avgpool(feats).flatten(1)

        elif self.backbone_name == 'regnet':
            if self.finetune_type == 'linear_prob' or self.finetune_type == 'svm':
                with torch.no_grad():
                    feats = self.backbone(batch)
            else:
                feats = self.backbone(batch)

            feats = feats.mean(-1)

        return feats

    def forward_head(self, feats):
        logits = {}

        logit = self.classifier(feats)
        if self.task_nums == 1:
            (task_name, num_classes), = self.task_info.items()
            logits[task_name] = logit
        else:
            logit = logit.view(-1, self.task_nums, 2)
            for index, task_name in enumerate(self.task_info):
                logits[task_name] = logit[:, index, :]

        return logits

    def forward(self, batch):
        feats = self.forward_features(batch)
        logits = self.forward_head(feats)

        return logits



def get_backbone(args):
    if args.ecg_model == 'resnet18':
        model = ResNet18(num_leads=args.num_leads)
    elif args.ecg_model == 'resnet34':
        model = ResNet34(num_leads=args.num_leads)
    elif args.ecg_model == 'resnet50':
        model = ResNet50(num_leads=args.num_leads)
    elif args.ecg_model == 'resnet101':
        model = ResNet101(num_leads=args.num_leads)
    elif args.ecg_model == 'regnet':
        from .regnet1d import Net1D 
        model = Net1D(
                in_channels=args.num_leads, 
                base_filters=64, #32 64
                ratio=1, 
                filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
                m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
                kernel_size=16, 
                stride=2, 
                groups_width=16,
                verbose=False, 
                use_bn=False,
                use_do=False,
                return_features=True,
                n_classes=1) 
    elif args.ecg_model == 'merl':
        model = ResNet18()

    # load pretrained ecg encoder
    if 'resnet' in args.ecg_model or args.ecg_model == 'regnet':
        checkpoint_path = getattr(args, 'pretrained_checkpoint', None)
        if checkpoint_path is None and args.pretrained_exp_name is not None:
            if args.pretrained_metric is None:
                raise ValueError(
                    "--pretrained_metric is required when --pretrained_exp_name is used."
                )
            experiment_dir = os.path.join(args.custom_pretrained_root, args.pretrained_exp_name)
            if not os.path.isdir(experiment_dir):
                raise FileNotFoundError(f"Pretraining experiment directory not found: {experiment_dir}")
            file_extension = args.pretrained_metric + '.pt'
            candidates = sorted(
                os.path.join(experiment_dir, file_name)
                for file_name in os.listdir(experiment_dir)
                if file_name.endswith(file_extension)
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No checkpoint ending with '{file_extension}' was found in {experiment_dir}."
                )
            checkpoint_path = candidates[0]

        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"ECG-CLIP checkpoint not found: {checkpoint_path}")
            # Pretraining checkpoints include the argparse Namespace in addition
            # to tensors, so PyTorch 2.6+ needs weights_only=False explicitly.
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_state_dict = checkpoint['model']
            ecg_encoder_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    k = k.removeprefix('module.')
                if k.startswith('ecg_encoder.'):
                    k = k.removeprefix('ecg_encoder.')
                    ecg_encoder_state_dict[k] = v

            msg = model.load_state_dict(ecg_encoder_state_dict, strict=False)
        
        else:
            msg = "Random initialized!"
    else:
        if args.ecg_model == 'merl':
            pth = getattr(args, 'merl_checkpoint', None)
            if not pth:
                raise ValueError("--merl_checkpoint is required when --ecg_model merl.")
            checkpoint = torch.load(pth, map_location="cpu", weights_only=False)
            msg = model.load_state_dict(checkpoint, strict=False)

    return model, msg


if __name__ == '__main__':
    print('hello world')
