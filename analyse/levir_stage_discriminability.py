"""LEVIR stage-discriminability preflight (Run7, read-only, no training).

Question: does the decoder final stage d1 (64x64, 160ch — the PBRU head input)
carry more linearly-separable change information than the Run2 trained head's
2-channel coarse projection? If yes, the fixed bilinear-x4 output head (not the
encoder/decoder) is the LEVIR small-object bottleneck, motivating the Run7
PixelShuffle/PBRU head.

Method (all at the decoder's native 64x64 scale, so head vs features are
comparable):
  - full linear bound: pooled-covariance (ridge) LDA on d1 fitted on the FIRST
    half of the test pairs, scored on the SECOND half (held-out) and reported
    in-sample for reference -> the best ANY linear readout of d1 can do @64;
  - diagonal (per-channel) LDA for reference, per-channel Fisher top-20,
    top-1 single-channel AUROC@64;
  - the trained head's change-probability AUROC@64 (pre-interp softmax);
  - the trained head's change-probability AUROC@256 (bilinear x4, deployed path).
The trained head IS a full linear map d1->2ch, so full-LDA >= head is the
correct comparison; full-LDA < head means the coarse scale is already saturated
and only sub-cell (phase) freedom remains — the exact case PBRU targets.
NOTE: with a FROZEN Run2 decoder this only tests STATIC coarse-scale headroom;
sub-cell phase freedom is learned end-to-end and is adjudicated by C0/C1/M1.

Labels: 1=change, 0=no-change, 255=ignore (cells touching 255 are excluded).
AUROC is computed with tie-corrected average ranks (Wilcoxon), no sklearn.

Usage (on server):
  python analyse/levir_stage_discriminability.py \
    --cfg <vssm yaml> --checkpoint <Run2 LEVIR best_F1 pth> \
    --dataset_root /share_datasets/CD/LEVIR-CD-256 \
    --test_list .../list/test.txt --output_dir <dir> [--max_samples 200] [--gpu 0]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from changedetection.configs.config import get_config
from changedetection.models.STRRepNet import STRRepNet
from changedetection.datasets.make_data_loader import ChangeDetectionDataset, read_list


def roc_auc(scores, labels):
    """Tie-corrected (average-rank) Mann-Whitney AUC on GPU tensors."""
    s = scores.double().reshape(-1)
    y = labels.reshape(-1)
    if s.numel() == 0 or (y == 1).sum().item() == 0 or (y == 0).sum().item() == 0:
        return float("nan")
    order = torch.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    n = s.numel()
    grp_start = torch.cat([torch.tensor([True], device=s.device),
                           s_sorted[1:] != s_sorted[:-1]])
    grp_ids = torch.cumsum(grp_start.long(), 0)
    counts = torch.bincount(grp_ids)
    starts = torch.cat([torch.tensor([0], device=s.device),
                        torch.cumsum(counts, 0)[:-1]])
    avg_rank = (starts.float() + (starts + counts - 1).float()) / 2.0 + 1.0
    ranks = avg_rank[grp_ids]
    n0 = int((y_sorted == 0).sum().item())
    n1 = int((y_sorted == 1).sum().item())
    r1 = ranks[y_sorted == 1].sum().item()
    return (r1 - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def build_model(args, config):
    v = config.MODEL.VSSM
    return STRRepNet(
        pretrained=None, rep_mode="full", dim=160, use_residual=True,
        encoder_train="last2", use_botr=False, use_nscr=False, nscr_scope="high2",
        head_mode="bilinear", use_pbru=False, pbru_upscale=4,
        patch_size=v.PATCH_SIZE, in_chans=v.IN_CHANS, num_classes=config.MODEL.NUM_CLASSES,
        depths=v.DEPTHS, dims=v.EMBED_DIM,
        ssm_d_state=v.SSM_D_STATE, ssm_ratio=v.SSM_RATIO, ssm_rank_ratio=v.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if v.SSM_DT_RANK == "auto" else int(v.SSM_DT_RANK)),
        ssm_act_layer=v.SSM_ACT_LAYER, ssm_conv=v.SSM_CONV, ssm_conv_bias=v.SSM_CONV_BIAS,
        ssm_drop_rate=v.SSM_DROP_RATE, ssm_init=v.SSM_INIT, forward_type=v.SSM_FORWARDTYPE,
        mlp_ratio=v.MLP_RATIO, mlp_act_layer=v.MLP_ACT_LAYER, mlp_drop_rate=v.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE, patch_norm=v.PATCH_NORM,
        norm_layer=v.NORM_LAYER, downsample_version=v.DOWNSAMPLE,
        patchembed_version=v.PATCHEMBED, gmlp=v.GMLP, use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )


def gt64_from_label(label):
    """(64,64) per-cell labels: 1 if any change pixel, valid=not touching 255."""
    gt = torch.from_numpy(label.astype(np.int64)).cuda()  # (256,256)
    g4 = gt.reshape(64, 4, 64, 4)
    chg = (g4 == 1).any(dim=1).any(dim=2)          # (64,64)
    ign = (g4 == 255).any(dim=1).any(dim=2)
    valid = ~ign
    return chg, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--test_list", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    for attr, val in [("opts", None), ("batch_size", 16), ("data_path", ""), ("zip", False),
                      ("cache_mode", None), ("pretrained", ""), ("resume", None),
                      ("accumulation_steps", None), ("use_checkpoint", False),
                      ("disable_amp", False), ("output", ""), ("tag", "")]:
        if not hasattr(args, attr):
            setattr(args, attr, val)

    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    config = get_config(args)

    model = build_model(args, config).cuda()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.eval()
    print("loaded", args.checkpoint)

    ds = ChangeDetectionDataset(args.dataset_root, read_list(args.test_list), 256, type="test")
    n = min(len(ds), args.max_samples)
    n_fit = n // 2

    # capture d1 (head input) + coarse logits (head output, pre-interp)
    d1_list, logit_list = [], []
    def fhook(m, i, o):
        d1_list.append(i[0].detach())
        logit_list.append(o.detach())
    hook = model.head.register_forward_hook(fhook)

    D = 160
    s0 = torch.zeros(D, device="cuda", dtype=torch.float64)
    s1 = torch.zeros(D, device="cuda", dtype=torch.float64)
    q0 = torch.zeros(D, device="cuda", dtype=torch.float64)
    q1 = torch.zeros(D, device="cuda", dtype=torch.float64)
    G0 = torch.zeros(D, D, device="cuda", dtype=torch.float64)
    G1 = torch.zeros(D, D, device="cuda", dtype=torch.float64)
    n0 = n1 = 0

    def forward_sample(i):
        a, b, label, _ = ds[i]
        a = torch.from_numpy(a).unsqueeze(0).cuda().float()
        b = torch.from_numpy(b).unsqueeze(0).cuda().float()
        with torch.no_grad():
            model(a, b)
        d1 = d1_list.pop()   # (1,160,64,64)
        lg = logit_list.pop()  # (1,2,64,64)
        chg, valid = gt64_from_label(label)
        return d1, lg, chg, valid

    print(f"[FIT] LDA on d1, first {n_fit} pairs")
    for i in range(n_fit):
        d1, lg, chg, valid = forward_sample(i)
        f = d1[0].permute(1, 2, 0).reshape(-1, D)[valid.reshape(-1)].double()
        y = chg[valid]
        m = (y == 1)
        n1 += int(m.sum().item())
        n0 += int((~m).sum().item())
        s1 += f[m].sum(0)
        s0 += f[~m].sum(0)
        q1 += (f[m] ** 2).sum(0)
        q0 += (f[~m] ** 2).sum(0)
        G1 += f[m].T @ f[m]
        G0 += f[~m].T @ f[~m]
    mu0 = s0 / n0
    mu1 = s1 / n1
    var0 = q0 / n0 - mu0 ** 2
    var1 = q1 / n1 - mu1 ** 2
    den = var0 + var1 + 1e-6
    w = (mu1 - mu0) / den
    fisher = ((mu1 - mu0) ** 2) / den
    # full pooled-covariance LDA (ridge): the best any linear readout of d1 can do
    S0 = G0 / n0 - torch.outer(mu0, mu0)
    S1 = G1 / n1 - torch.outer(mu1, mu1)
    Sp = (n0 * S0 + n1 * S1) / (n0 + n1)
    lam = 1e-3 * torch.diagonal(Sp).mean()
    w_full = torch.linalg.solve(Sp + lam * torch.eye(D, device="cuda", dtype=torch.float64),
                                mu1 - mu0)
    topk = torch.topk(fisher, 20)
    top_channels = [{"ch": int(c), "fisher": float(v)} for c, v in zip(topk.indices, topk.values)]
    print("  top-5 Fisher channels:", [(int(c), round(float(v), 3)) for c, v in zip(topk.indices[:5], topk.values[:5])])

    # scoring: second half held-out (+ in-sample for reference)
    def score_split(lo, hi, tag):
        s_lda, s_ldaf, s_top1, s_p64, s_p256 = [], [], [], [], []
        y64s, y256s = [], []
        for i in range(lo, hi):
            d1, lg, chg, valid = forward_sample(i)
            f = d1[0].permute(1, 2, 0).reshape(-1, D)[valid.reshape(-1)]
            y64 = chg[valid].bool()
            fd = f.double()
            s_lda.append(fd @ w)
            s_ldaf.append(fd @ w_full)
            s_top1.append(f[:, topk.indices[0]])
            p = F.softmax(lg, dim=1)
            s_p64.append(p[0, 1][valid])
            p256 = F.interpolate(lg, size=(256, 256), mode="bilinear", align_corners=False)
            gt = torch.from_numpy(ds[i][2].astype(np.int64)).cuda()
            yv = (gt != 255).reshape(-1)
            s_p256.append(p256[0, 1].reshape(-1)[yv])
            y256s.append((gt == 1).reshape(-1)[yv])
            y64s.append(y64)
        a_lda = roc_auc(torch.cat(s_lda), torch.cat(y64s))
        a_ldaf = roc_auc(torch.cat(s_ldaf), torch.cat(y64s))
        a_top1 = roc_auc(torch.cat(s_top1), torch.cat(y64s))
        a_p64 = roc_auc(torch.cat(s_p64), torch.cat(y64s))
        a_p256 = roc_auc(torch.cat(s_p256), torch.cat(y256s))
        print(f"  [{tag}] d1-fullLDA@64={a_ldaf:.4f} d1-diagLDA@64={a_lda:.4f} "
              f"top1ch@64={a_top1:.4f} head@64={a_p64:.4f} head@256={a_p256:.4f}")
        return a_ldaf, a_lda, a_top1, a_p64, a_p256

    print(f"[EVAL] held-out pairs {n_fit}..{n}")
    a_ldaf_ho, a_lda_ho, a_top1_ho, a_p64_ho, a_p256_ho = score_split(n_fit, n, "held-out")
    a_ldaf_is, a_lda_is, a_top1_is, a_p64_is, a_p256_is = score_split(0, n_fit, "in-sample")
    hook.remove()

    out = {
        "checkpoint": args.checkpoint,
        "n_pairs": n, "n_fit": n_fit,
        "lda_top_channels": top_channels,
        "held_out": {"d1_fullLDA_AUROC_64": round(a_ldaf_ho, 4),
                     "d1_diagLDA_AUROC_64": round(a_lda_ho, 4),
                     "top1_channel_AUROC_64": round(a_top1_ho, 4),
                     "head_AUROC_64": round(a_p64_ho, 4),
                     "head_AUROC_256": round(a_p256_ho, 4),
                     "coarse_linear_headroom_64": round(a_ldaf_ho - a_p64_ho, 4)},
        "in_sample": {"d1_fullLDA_AUROC_64": round(a_ldaf_is, 4),
                      "d1_diagLDA_AUROC_64": round(a_lda_is, 4),
                      "top1_channel_AUROC_64": round(a_top1_is, 4),
                      "head_AUROC_64": round(a_p64_is, 4),
                      "head_AUROC_256": round(a_p256_is, 4)},
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "stage_discriminability.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
