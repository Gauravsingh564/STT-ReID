import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .layers import GeM, TemporalAttention, ArcFace

class VideoReIDModel(nn.Module):
    def __init__(self, num_classes, swin_type="swin_base_patch4_window7_224", seq_len=8):
        super().__init__()
        self.backbone = timm.create_model(swin_type, pretrained=True)
        self.feat_dim = self.backbone.num_features
        self.backbone.avgpool = GeM()
        self.backbone.reset_classifier(0)
        self.backbone.set_grad_checkpointing(True)

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, self.feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim, nhead=8, dim_feedforward=self.feat_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_pool = TemporalAttention(self.feat_dim, hidden_dim=512)
        
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        
        self.classifier = ArcFace(in_features=self.feat_dim, out_features=num_classes, s=30.0, m=0.3)

    def forward(self, x, labels=None):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)

        frame_feats = self.backbone(x)
        frame_feats = frame_feats.view(B, T, -1)

        frame_feats = frame_feats + self.pos_embed
        transformer_feats = self.temporal_transformer(frame_feats)

        pooled_feat = self.temporal_pool(transformer_feats)
        feat_bn = self.bottleneck(pooled_feat)

        if self.training and labels is not None:
            penalty_logits, raw_logits = self.classifier(feat_bn, labels)  
            norm_feat = F.normalize(pooled_feat, p=2, dim=1)
            return feat_bn, penalty_logits, raw_logits, norm_feat
        else:
            return F.normalize(feat_bn, p=2, dim=1)
