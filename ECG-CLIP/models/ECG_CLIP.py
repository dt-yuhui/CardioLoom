from cgi import test
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from .resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from .vit1d import vit_base, vit_small, vit_tiny, vit_middle
from .regnet1d import Net1D


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, spacial_dim + 1, embed_dim) / embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)        
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(0, 2, 1) # convert X shape (B, C, L) to (B, L, C)

        self.cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
        self.cls_tokens = self.cls_tokens.expand(x.shape[0], -1, -1) 
        x = torch.cat((self.cls_tokens, x), dim=1)
        x = x + self.positional_embedding[:, :, :].to(x.dtype)  # (L+1)NC
        x, att_map = self.mhsa(x[:, :1, :], x, x, average_attn_weights=True)
        x = self.c_proj(x)
        return x.squeeze(0), att_map[:, :, 1:]
    
class ECGCLIP(torch.nn.Module):
    def __init__(self, args):
        super(ECGCLIP, self).__init__()
        
        self.proj_hidden = args.mlp_hidden_size
        self.proj_out = args.projection_size

        # # ecg signal encoder
        self.ecg_model = args.ecg_model
        self.num_leads = args.num_leads

        if 'resnet' in self.ecg_model:
            self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                embed_dim=self.proj_out, 
                                                num_heads=4, 
                                                output_dim=self.proj_out)
            if self.ecg_model == 'resnet18':
                model = ResNet18(self.num_leads)
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1)
            elif self.ecg_model == 'resnet34':
                model = ResNet34(self.num_leads)
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1)
            elif self.ecg_model == 'resnet50':
                model = ResNet50(self.num_leads)
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
            elif self.ecg_model == 'resnet101':
                model = ResNet101(self.num_leads)
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)

            self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)

        if 'vit' in self.ecg_model:
            if self.ecg_model == 'vit_tiny':
                model = vit_tiny(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_small':
                model = vit_small(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_middle':
                model = vit_middle(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_base':
                model = vit_base(num_leads=self.num_leads)
            self.proj_e_input = model.width    
            self.proj_e = nn.Sequential(
                nn.Linear(self.proj_e_input, self.proj_hidden),
                nn.BatchNorm1d(self.proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden, self.proj_out),
                nn.BatchNorm1d(self.proj_out),
            )
            self.linear1 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)

        if 'regnet' in self.ecg_model:
            model = Net1D(
                in_channels=self.num_leads, 
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
            self.downconv = nn.Conv1d(in_channels=1024, out_channels=self.proj_out, kernel_size=1)
            self.att_pool_head = AttentionPool2d(spacial_dim=20,
                                    embed_dim=self.proj_out, 
                                    num_heads=4, 
                                    output_dim=self.proj_out)
            self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)

        self.ecg_encoder = model
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        # text encoder
        pretrained_text_ckpt = args.text_model
        local_files_only = getattr(args, 'text_model_local_files_only', False)
        try:
            self.lm_model = AutoModel.from_pretrained(
                pretrained_text_ckpt,
                local_files_only=local_files_only,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_text_ckpt,
                local_files_only=local_files_only,
            )
        except OSError as exc:
            mode = "local-only" if local_files_only else "online/cache"
            raise RuntimeError(
                f"Unable to load text encoder '{pretrained_text_ckpt}' in {mode} mode. "
                "For an offline server, copy the complete Hugging Face model directory "
                "to the server, pass that directory with --text_model, and set "
                "--text_model_local_files_only True."
            ) from exc

        # for k, v in self.lm_model.named_parameters():
        #     print(k)
        
        # text projector
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )
        
        
    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=256,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output
    
    # @torch.no_grad()
    def ext_ecg_emb(self, ecg):

        if 'resnet' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            ecg_emb = self.downconv(ecg_emb)
            proj_ecg_emb, att_map = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

        if 'vit' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)

        return proj_ecg_emb
    
    # @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        # text_emb = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_emb = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return text_emb
    
    def forward(self, ecg, input_ids, attention_mask):
        ecg_emb = self.ecg_encoder(ecg)

        if 'resnet' in self.ecg_model or 'regnet' in self.ecg_model:
            # attention pooling (only for resnet/regnet models)
            ecg_emb = self.downconv(ecg_emb)
            proj_ecg_emb, _ = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

            ecg_emb = self.avgpool(ecg_emb).view(ecg_emb.shape[0], -1)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))
        
        if 'vit' in self.ecg_model:
            proj_ecg_emb = self.proj_e(ecg_emb)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))

        proj_ecg_emb = normalize(proj_ecg_emb, dim=-1)

        # get text feature
        # text feature extraction is independent of the type of ecg encoder
        text_emb = self.get_text_emb(input_ids, attention_mask)
        proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = normalize(proj_text_emb, dim=-1)

        return {
            'ecg_emb': [ecg_emb1, ecg_emb2],
            'proj_ecg_emb': proj_ecg_emb,
            'proj_text_emb': proj_text_emb
            }
