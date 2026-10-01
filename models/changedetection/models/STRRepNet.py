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

use_botr : Bi-Order Temporal Re-parameterization (Run5). Adds a zero-init
    reverse-concat [Q,P] branch to every TAR scale; folded back into the single
    temporal 1x1 at deploy via channel permutation (+0 deploy Params/FLOPs).

use_nscr : Native-Scale Commutative Re-parameterized Fusion (Run6). Adds
    zero-init native-scale BN branches to DCR fuse1+fuse2, absorbed into the
    existing cross-scale 1x1 at deploy via affine-interpolation commutation
    (+0 deploy Params/FLOPs). nscr_scope in {"high2","lonly","honly"}.

head_mode in {"bilinear", "pixelshuffle"} (Run7):
    bilinear    : head Conv2d(dim,2,1) at 1/4 scale + bilinear x4 (Run2-Run6 default)
    pixelshuffle: PBRUHead projecting dim -> 2*r^2 with PixelShuffle(r) to full res.
use_pbru : add the four zero-init phase-basis BN branches (coarse/u/v/uv) inside
    PBRUHead at train time; folded analytically into the single 1x1 projection at
    deploy (+0 deploy Params/FLOPs vs plain PixelShuffle, +0 vs Run2 after D* search).
pbru_upscale : r of PixelShuffle (fixed 4; 64x64 -> 256x256).

use_mpcr (Run8): refine.pw becomes MPCRPW1x1 — two zero-init grouped-1x1 BN
    branches (identity + identity|interleaved partition) absorbed into the original
    dense PW at deploy via P^-1 G P (+0 deploy Params/FLOPs). mpcr_mode in
    {"same2", "multi2"}, mpcr_groups fixed 4.

use_biftr (Run9): BiFTR — wraps the frozen stage2->trainable stage3 downsample
    Conv (192->384, stride 2; actual config v3: k3 s2 p1) as
    y = (I+d_out) W ((I+d_in) x). d_in/d_out are zero-init 1x1 convs
    (train-only; base Conv stays frozen). Deploy folds W_eq = A W B, b_eq = A b
    back into the single original Conv (+0 deploy). biftr_mode in {"post","bi"}:
    post = d_out only (C0_FTR_Post), bi = both (M1_BiFTR).

use_pfdr (Run10): PFDR — Pre-Fusion Dilated Re-parameterization. t1 passes
    through a PFDRDW5 (mode plain|rep) BEFORE DCR fuse1; training adds zero-init
    DW3 / DW3(d=2) / DW1 depthwise bases that fold analytically into ONE deploy
    DW5x5 (FP64, one FP32 cast). Budget is reclaimed via the D* decoder-width
    search (doc §4.2). The PFDRDW5 module is attached AFTER all other
    RNG-consuming constructions so C0/M1 epoch-0 outputs stay bitwise identical
    (Run9-style RNG discipline, doc §5.7).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from changedetection.models.Mamba_backbone import Backbone_VSSM, install_biftr_transition
from changedetection.models.tar import MultiScaleTAR
from changedetection.models.dcr_decoder import DCRDecoder
from changedetection.models.reparam import PBRUHead, PFDRDW5


class STRRepNet(nn.Module):
    def __init__(self, pretrained=None, rep_mode="full", dim=160, use_residual=True,
                 encoder_train="frozen", use_botr=False, use_nscr=False,
                 nscr_scope="high2", head_mode="bilinear", use_pbru=False,
                 pbru_upscale=4, use_mpcr=False, mpcr_mode="multi2", mpcr_groups=4,
                 use_biftr=False, biftr_mode="bi", use_pfdr=False, pfdr_mode="rep",
                 **encoder_kwargs):
        super().__init__()
        self.rep_mode = rep_mode
        self.dim = dim
        self.use_residual = use_residual
        self.encoder_train = encoder_train
        self.use_botr = use_botr
        self.use_nscr = use_nscr
        self.nscr_scope = nscr_scope if use_nscr else "none"
        self.head_mode = head_mode
        self.use_pbru = use_pbru
        self.pbru_upscale = int(pbru_upscale)
        self.use_mpcr = use_mpcr
        self.mpcr_mode = mpcr_mode
        self.mpcr_groups = int(mpcr_groups)
        self.use_biftr = use_biftr
        self.biftr_mode = biftr_mode
        self.biftr = None
        self.use_pfdr = use_pfdr
        self.pfdr_mode = pfdr_mode

        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **encoder_kwargs)
        self._setup_encoder_train()

        use_temporal_aux = rep_mode in ("tar", "full")
        use_dcr_aux = rep_mode in ("dcr", "full")

        self.tar = MultiScaleTAR(
            encoder_dims=self.encoder.dims, dim=dim,
            use_temporal_aux=use_temporal_aux, use_dcr_aux=use_dcr_aux,
            use_residual=use_residual, use_reverse_aux=use_botr,
        )
        self.decoder = DCRDecoder(dim=dim, use_aux=use_dcr_aux, use_residual=use_residual,
                                  nscr_scope=self.nscr_scope, use_mpcr=use_mpcr,
                                  mpcr_mode=mpcr_mode, mpcr_groups=mpcr_groups,
                                  use_pfdr=use_pfdr, pfdr_mode=pfdr_mode)
        if head_mode == "pixelshuffle":
            self.head = PBRUHead(dim, num_classes=2, upscale=self.pbru_upscale,
                                 use_phase_rep=use_pbru)
        else:
            self.head = nn.Conv2d(dim, 2, 1)

        # Run9 BiFTR: install AFTER all RNG-consuming constructions. nn.Conv2d
        # creation draws RNG for its (immediately zeroed) init, so installing
        # earlier would shift the shared-init RNG stream and break the epoch-0
        # bitwise identity with use_biftr=0. The wrapper's home layers[1] is
        # frozen by last2; the train-only deltas are re-enabled here while the
        # base Conv stays frozen.
        if use_biftr:
            self.biftr = install_biftr_transition(self.encoder, mode=biftr_mode,
                                                  transition="stage2_to_stage3")
            self.biftr.conv.requires_grad_(False)
            if hasattr(self.biftr, "d_in"):
                self.biftr.d_in.requires_grad_(True)
            if hasattr(self.biftr, "d_out"):
                self.biftr.d_out.requires_grad_(True)

        # Run10 PFDR: attach AFTER the head for the same RNG reason. mode="rep"
        # constructs extra zero-init aux convs (which consume RNG); installing
        # after every other construction keeps C0 (plain) / M1 (rep) / A0 (off)
        # downstream/shared weights bitwise identical at epoch 0.
        if use_pfdr:
            self.decoder.prefuse1 = PFDRDW5(dim, mode=pfdr_mode, use_alpha=True, deploy=False)

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

    def forward(self, pre, post):
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
        if self.head_mode != "pixelshuffle":
            logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_train == "frozen":
            self.encoder.eval()  # keep encoder as a frozen feature extractor
        return self

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.biftr is not None:
            self.biftr.switch_to_deploy()
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        if isinstance(self.head, PBRUHead):
            self.head.switch_to_deploy()
        return self
