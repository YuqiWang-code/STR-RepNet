"""STR-RepNet: Frozen VMamba-Tiny Siamese Encoder + TAR bridge + DCR decoder.

rep_mode in {"plain", "tar", "full"}:
  plain : single-path counterpart (no auxiliary rep branches)
  tar   : temporal sum/signed-diff branches only
  full  : temporal + decoder-wide (spatial/channel/cross-scale) rep branches
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from changedetection.models.Mamba_backbone import Backbone_VSSM
from changedetection.models.tar import MultiScaleTAR
from changedetection.models.dcr_decoder import DCRDecoder


class STRRepNet(nn.Module):
    def __init__(self, pretrained=None, rep_mode="full", dim=160, **encoder_kwargs):
        super().__init__()
        self.rep_mode = rep_mode
        self.dim = dim

        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **encoder_kwargs)
        # freeze the encoder completely
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        use_temporal_aux = rep_mode in ("tar", "full")
        use_dcr_aux = rep_mode == "full"

        self.tar = MultiScaleTAR(
            encoder_dims=self.encoder.dims, dim=dim,
            use_temporal_aux=use_temporal_aux, use_dcr_aux=use_dcr_aux,
        )
        self.decoder = DCRDecoder(dim=dim, use_aux=use_dcr_aux)
        self.head = nn.Conv2d(dim, 2, 1)

    def forward(self, pre, post):
        with torch.no_grad():
            pre_feats = self.encoder(pre)
            post_feats = self.encoder(post)
        feats = self.tar(pre_feats, post_feats)
        x = self.decoder(feats)
        logits = self.head(x)
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def train(self, mode=True):
        super().train(mode)
        # Frozen feature extractor: keep encoder in eval regardless of decoder mode.
        self.encoder.eval()
        return self

    @torch.no_grad()
    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self
