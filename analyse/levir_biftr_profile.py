"""Run9 BiFTR mechanism profile (read-only; never used for checkpoint selection).

Hooks five encoder points around the frozen->trainable stage2/3 transition:
    pre_transition       (stage2 blocks out, 32x32, 192ch)
    transition_pre_norm  (downsample conv / BiFTR wrapper out, 16x16, 384ch)
    transition_post_norm (downsample norm out, 16x16, 384ch)
    stage3_out           (stage3 blocks out, 16x16, 384ch)
    stage4_out           (stage4 blocks out, 8x8, 768ch)
and reports per stage: centroid distance (raw + normalized), Fisher (top-10),
diag-LDA AUROC (fit first half / held-out second half), covariance effective
rank, mean |inter-channel correlation|. For BiFTR models it also records the
weight-space stats (||d_in||F, ||d_out||F, ||W_eq-W||F/||W||F, cross-term ratio).

Usage:
    python analyse/levir_biftr_profile.py --cfg <vssm yaml> \
        --checkpoint <pth> --dataset_root /share_datasets/CD/LEVIR-CD-256 \
        --test_list .../list/test.txt --output_dir <dir> \
        [--use_biftr 0] [--biftr_mode bi] [--max_samples 200] [--gpu 0]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from changedetection.configs.config import get_config
from changedetection.models.STRRepNet import STRRepNet
from changedetection.datasets.make_data_loader import ChangeDetectionDataset, read_list
from levir_stage_discriminability import roc_auc

STAGES = [
    ("pre_transition", ["layers", 1, "blocks"], 32, 192),
    ("transition_pre_norm", ["layers", 1, "downsample", 1], 16, 384),
    ("transition_post_norm", ["layers", 1, "downsample", 3], 16, 384),
    ("stage3_out", ["layers", 2, "blocks"], 16, 384),
    ("stage4_out", ["layers", 3, "blocks"], 8, 768),
]


def gt_at(label, scale):
    """(scale, scale) any-change cells; valid = not touching a 255 pixel."""
    s = 256 // scale
    gt = torch.from_numpy(label.astype(np.int64)).cuda()
    g = gt.reshape(scale, s, scale, s)
    chg = (g == 1).any(dim=1).any(dim=2)
    ign = (g == 255).any(dim=1).any(dim=2)
    return chg, ~ign


def build_model(args, config):
    v = config.MODEL.VSSM
    return STRRepNet(
        pretrained=None, rep_mode="full", dim=160, use_residual=True,
        encoder_train="last2", use_botr=False, use_nscr=False, nscr_scope="high2",
        head_mode="bilinear", use_pbru=False, pbru_upscale=4,
        use_mpcr=False, mpcr_mode="multi2", mpcr_groups=4,
        use_biftr=bool(args.use_biftr), biftr_mode=args.biftr_mode,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--test_list", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--use_biftr", type=int, default=0)
    ap.add_argument("--biftr_mode", type=str, default="bi", choices=["post", "bi"])
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

    captures = {}
    hooks = []
    for name, path, scale, _D in STAGES:
        mod = model.encoder
        for part in path:
            mod = mod[part]
        def make_hook(k):
            def h(m, i, o):
                captures[k] = o.detach()
            return h
        hooks.append(mod.register_forward_hook(make_hook(name)))

    ds = ChangeDetectionDataset(args.dataset_root, read_list(args.test_list), 256, type="test")
    n = min(len(ds), args.max_samples)
    n_fit = n // 2

    def forward_sample(i):
        a, b, label, _ = ds[i]
        a = torch.from_numpy(a).unsqueeze(0).cuda().float()
        b = torch.from_numpy(b).unsqueeze(0).cuda().float()
        with torch.no_grad():
            model(a, b)
        return {name: captures[name] for name, _, _, _ in STAGES}, label

    fits = {}
    for name, _, scale, D in STAGES:
        fits[name] = {"D": D, "scale": scale,
                      "s0": torch.zeros(D, device="cuda", dtype=torch.float64),
                      "s1": torch.zeros(D, device="cuda", dtype=torch.float64),
                      "q0": torch.zeros(D, device="cuda", dtype=torch.float64),
                      "q1": torch.zeros(D, device="cuda", dtype=torch.float64),
                      "G": torch.zeros(D, D, device="cuda", dtype=torch.float64),
                      "n0": 0, "n1": 0}

    for i in range(n_fit):
        outs, label = forward_sample(i)
        for name, _, scale, _D in STAGES:
            chg, valid = gt_at(label, scale)
            f = outs[name][0].permute(1, 2, 0).reshape(-1, fits[name]["D"])[valid.reshape(-1)].double()
            y = chg[valid]
            m = (y == 1)
            st = fits[name]
            st["n1"] += int(m.sum().item())
            st["n0"] += int((~m).sum().item())
            st["s1"] += f[m].sum(0)
            st["s0"] += f[~m].sum(0)
            st["q1"] += (f[m] ** 2).sum(0)
            st["q0"] += (f[~m] ** 2).sum(0)
            st["G"] += f.T @ f

    w_lda = {}
    report = {}
    for name, _, scale, _D in STAGES:
        st = fits[name]
        D = st["D"]
        mu0 = st["s0"] / st["n0"]
        mu1 = st["s1"] / st["n1"]
        var0 = st["q0"] / st["n0"] - mu0 ** 2
        var1 = st["q1"] / st["n1"] - mu1 ** 2
        den = var0 + var1 + 1e-6
        w_lda[name] = (mu1 - mu0) / den
        fisher = ((mu1 - mu0) ** 2) / den
        topk = torch.topk(fisher, 10)
        tot = st["n0"] + st["n1"]
        mu_all = (st["s0"] + st["s1"]) / tot
        cov = st["G"] / tot - torch.outer(mu_all, mu_all)
        ev = torch.linalg.eigvalsh(cov)
        ev = torch.clamp(ev, min=1e-12)
        p = ev / ev.sum()
        ent = -(p * torch.log(p + 1e-12)).sum()
        dg = torch.sqrt(torch.diag(cov) + 1e-12)
        corr = cov / torch.outer(dg, dg)
        off = corr[~torch.eye(D, dtype=torch.bool, device=cov.device)]
        report[name] = {
            "scale": scale,
            "centroid_l2": float((mu1 - mu0).norm().item()),
            "centroid_l2_norm": float(((mu1 - mu0).norm() / (mu1.norm() + mu0.norm() + 1e-12)).item()),
            "fisher_mean": float(fisher.mean().item()),
            "fisher_top10": [{"ch": int(c), "fisher": float(v)} for c, v in zip(topk.indices, topk.values)],
            "eff_rank": round(float(torch.exp(ent).item()), 2),
            "mean_abs_corr": round(float(off.abs().mean().item()), 4),
        }

    def score_split(lo, hi):
        acc = {name: {"s": [], "y": []} for name, _, _, _ in STAGES}
        for i in range(lo, hi):
            outs, label = forward_sample(i)
            for name, _, scale, _D in STAGES:
                chg, valid = gt_at(label, scale)
                f = outs[name][0].permute(1, 2, 0).reshape(-1, fits[name]["D"])[valid.reshape(-1)]
                acc[name]["s"].append(f.double() @ w_lda[name])
                acc[name]["y"].append(chg[valid].bool())
        return {name: roc_auc(torch.cat(acc[name]["s"]), torch.cat(acc[name]["y"]))
                for name, _, _, _ in STAGES}

    auc_ho = score_split(n_fit, n)
    auc_is = score_split(0, n_fit)
    for name, _, _, _ in STAGES:
        report[name]["diag_lda_auc_ho"] = round(auc_ho[name], 4)
        report[name]["diag_lda_auc_is"] = round(auc_is[name], 4)

    for h_ in hooks:
        h_.remove()

    out = {"checkpoint": args.checkpoint, "n_pairs": n, "n_fit": n_fit,
           "use_biftr": args.use_biftr, "biftr_mode": args.biftr_mode, "stages": report}
    if args.use_biftr:
        st = model.biftr.branch_stats()
        out["weight_space"] = {k: round(v, 6) if isinstance(v, float) else v
                               for k, v in st.items()}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "biftr_profile.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
