"""STR-RepNet: Frozen/partially-tunable VMamba-Tiny Siamese Encoder + TAR + DCR decoder.

rep_mode in {"plain", "tar", "dcr", "full"} (2x2 ablation):
  plain : no temporal aux, no DCR aux
  tar   : temporal aux only
  dcr   : DCR aux only
  full  : temporal + DCR aux

encoder_train in {"frozen", "last2", "full"}:
  frozen : freeze the whole encoder (current default)
  last2  : freeze stage1+stage2, unfreeze stage3+stage4 (+ outnorm2/3)
  full   : unfreeze the whole encoder (capacity upper bound)

use_residual : foldable clean residual `+ alpha*x` inside each linear op (BN-FR).

use_boundary_aux : training-only inner-boundary auxiliary supervision (IBAS, Run4).
    Adds a zero-init Conv2d(dim,1,1) boundary head on the final decoder feature;
    it is DELETED in switch_to_deploy() (main path never reads it, +0 deploy cost).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from changedetection.models.Mamba_backbone import Backbone_VSSM
from changedetection.models.tar import MultiScaleTAR
from changedetection.models.dcr_decoder import DCRDecoder


class STRRepNet(nn.Module):
    def __init__(self, pretrained=None, rep_mode="full", dim=160, use_residual=True,
                 use_edge=False, encoder_train="frozen", use_boundary_aux=False, **encoder_kwargs):
        super().__init__()
        self.rep_mode = rep_mode
        self.dim = dim
        self.use_residual = use_residual
        self.use_edge = use_edge
        self.encoder_train = encoder_train
        self.use_boundary_aux = use_boundary_aux

        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **encoder_kwargs)
        self._setup_encoder_train()

        use_temporal_aux = rep_mode in ("tar", "full")
        use_dcr_aux = rep_mode in ("dcr", "full")

        self.tar = MultiScaleTAR(
            encoder_dims=self.encoder.dims, dim=dim,
            use_temporal_aux=use_temporal_aux, use_dcr_aux=use_dcr_aux,
            use_residual=use_residual,
        )
        self.decoder = DCRDecoder(dim=dim, use_aux=use_dcr_aux, use_residual=use_residual, use_edge=use_edge)
        self.head = nn.Conv2d(dim, 2, 1)

        # training-only boundary aux head (IBAS): zero-init, deploy-deleted
        if self.use_boundary_aux:
            self.boundary_head = nn.Conv2d(dim, 1, 1)
            nn.init.zeros_(self.boundary_head.weight)
            nn.init.zeros_(self.boundary_head.bias)
        else:
            self.boundary_head = None

    def _setup_encoder_train(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if self.encoder_train == "last2":
            # unfreeze stage3 + stage4 (+ their outnorm)
            for i in (2, 3):
                for p in self.encoder.layers[i].parameters():
                    p.requires_grad_(True)
                outnorm = getattr(self.encoder, f"outnorm{i}")
                for p in outnorm.parameters():
                    p.requires_grad_(True)
        elif self.encoder_train == "full":
            for p in self.encoder.parameters():
                p.requires_grad_(True)

    def forward(self, pre, post, return_aux=False):
        if self.encoder_train == "frozen":
            with torch.no_grad():
                pre_feats = self.encoder(pre)
                post_feats = self.encoder(post)
        else:
            pre_feats = self.encoder(pre)
            post_feats = self.encoder(post)
        feats = self.tar(pre_feats, post_feats)
        x = self.decoder(feats)
        logits = self.head(x)
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        if return_aux and self.boundary_head is not None:
            boundary = self.boundary_head(x)
            boundary = F.interpolate(boundary, size=pre.shape[-2:], mode="bilinear", align_corners=False)
            return logits, boundary
        return logits

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_train == "frozen":
            self.encoder.eval()  # keep encoder as a frozen feature extractor
        return self

    def remove_boundary_aux(self):
        """Delete the training-only boundary head; the main path never read it."""
        self.boundary_head = None
        self.use_boundary_aux = False
        return self

    @torch.no_grad()
    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        if self.boundary_head is not None:
            self.boundary_head = None
        self.use_boundary_aux = False
        return self
