"""Run10 PFDR Phase -1 diagnostic: LEVIR fine-lateral retention profile
(mechanism-only, READ-ONLY on the Run2 anchor checkpoint; no checkpoint
selection, no hyper-parameter choice).

Question (doc §6.2): does the fine lateral t1 carry MORE small-change
separability than the stages AFTER the cross-scale fusion? If held-out
separability drops from t1 -> fuse1 -> block1 -> refine, it supports the
"pre-fusion detail dilution" mechanism hypothesis behind PFDR. If not, the
prior is downgraded to exploratory (Run10 still runs; doc §6.2).

Hooked stages (all at the decoder's native 64x64 scale, D=160 channels):
    t1         : TAR stage1 output (decoder input; forward_pre_hook)
    fuse1_out  : fuse1 output pre-act (the cross-scale semantic mix)
    block1_out : block1 output
    refine_out : decoder d1 (head input)

Per small/medium/large bin (component-size bins from TRAIN-GT quantiles, the
Run6 small<=502px partition procedure) and per stage, on changed-vs-unchanged
64x64 cells (change cell = any GT change pixel in the 4x4 block; bin = the
dominant GT component covering the cell):
    centroid L2 distance (raw + normalized), per-channel Fisher mean + top-5,
    diagonal-LDA AUROC (fit first half / held-out second half), covariance
    effective rank, mean |inter-channel correlation|.

Usage (on server):
    python analyse/levir_prefusion_retention_profile.py \
        --cfg <vssm yaml> --checkpoint <Run2 LEVIR best_F1 pth> \
        --dataset_root /share_datasets/CD/LEVIR-CD-256 \
        --train_list .../list/train.txt --test_list .../list/test.txt \
        --output_dir <dir> [--max_samples 200] [--gpu 0]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage as ndi
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from changedetection.configs.config import get_config
from changedetection.models.STRRepNet import STRRepNet
from changedetection.datasets.make_data_loader import ChangeDetectionDataset, read_list
from levir_stage_discriminability import roc_auc

STAGES = ["t1", "fuse1_out", "block1_out", "refine_out"]
BIN_NAMES = {1: "small", 2: "medium", 3: "large"}


def build_model(args, config):
    v = config.MODEL.VSSM
    return STRRepNet(
        pretrained=None, rep_mode="full", dim=160, use_residual=True,
        encoder_train="last2", use_botr=False, use_nscr=False, nscr_scope="high2",
        head_mode="bilinear", use_pbru=False, pbru_upscale=4,
        use_mpcr=False, mpcr_mode="multi2", mpcr_groups=4,
        use_biftr=False, biftr_mode="bi", use_pfdr=False, pfdr_mode="rep",
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


def train_bin_edges(dataset_root, train_list, max_train=400):
    """small/medium/large edges from TRAIN-GT component areas (Run6 procedure)."""
    ds = ChangeDetectionDataset(dataset_root, read_list(train_list), 256, type="test")
    areas = []
    n_tr = len(ds)
    idx = np.random.RandomState(0).choice(n_tr, min(max_train, n_tr), replace=False)
    for i in idx:
        _, _, lab, _ = ds[i]
        mask = (lab == 1).astype(np.uint8)
        if mask.sum() == 0:
            continue
        lbl, ncomp = ndi.label(mask, structure=np.ones((3, 3)))
        for c in range(1, ncomp + 1):
            areas.append(int((lbl == c).sum()))
    areas = np.array(areas, dtype=np.float64)
    if areas.size == 0:
        return 502.0, 1500.0, 0
    q33, q67 = np.quantile(areas, [0.33, 0.67])
    return float(q33), float(q67), int(areas.size)


def cell_bins(label, q33, q67):
    """(256,256) GT label -> (64,64) cell bins: -1 ignore, 0 unchanged, 1..3 small..large."""
    gt = label.astype(np.int64)
    mask = (gt == 1).astype(np.uint8)
    bins = np.full((64, 64), -1, dtype=np.int8)
    ign = (gt == 255)
    g4_ign = ign.reshape(64, 4, 64, 4).any(axis=1).any(axis=2)
    if mask.sum() == 0:
        bins[~g4_ign] = 0
        return bins
    lbl, ncomp = ndi.label(mask, structure=np.ones((3, 3)))
    comp_area = np.bincount(lbl.reshape(-1), minlength=ncomp + 1)[1:]
    comp_bin = np.zeros(ncomp + 1, dtype=np.int8)
    for c in range(1, ncomp + 1):
        a = comp_area[c - 1]
        comp_bin[c] = 0 if a <= q33 else (1 if a <= q67 else 2)
    for i in range(64):
        for j in range(64):
            patch = lbl[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            if patch.max() == 0:
                bins[i, j] = -1 if g4_ign[i, j] else 0
            else:
                ids, counts = np.unique(patch[patch > 0], return_counts=True)
                dom = ids[int(np.argmax(counts))]
                bins[i, j] = comp_bin[dom] + 1
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--train_list", type=str, required=True)
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

    if not HAS_SCIPY:
        print("[FAIL] scipy not available; component binning cannot run")
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    config = get_config(args)

    q33, q67, n_train_comp = train_bin_edges(args.dataset_root, args.train_list)
    print(f"[BINS] train-GT component quantiles: small<={q33:.0f}px, medium<={q67:.0f}px "
          f"(n={n_train_comp} components)")

    model = build_model(args, config).cuda()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.eval()
    print("loaded", args.checkpoint)

    # hooks: t1 via decoder input; fuse1/block1/refine via outputs
    caps = {}
    def pre_dec(m, i):
        caps["t1"] = i[0][0].detach()
    def make_hook(k):
        def h(m, i, o):
            caps[k] = o.detach()
        return h
    hooks = [model.decoder.register_forward_pre_hook(pre_dec)]
    hooks += [model.decoder.fuse1.register_forward_hook(make_hook("fuse1_out"))]
    hooks += [model.decoder.block1.register_forward_hook(make_hook("block1_out"))]
    hooks += [model.decoder.refine.register_forward_hook(make_hook("refine_out"))]

    ds = ChangeDetectionDataset(args.dataset_root, read_list(args.test_list), 256, type="test")
    n = min(len(ds), args.max_samples)
    n_fit = n // 2
    D = 160

    def forward_sample(i):
        a, b, label, _ = ds[i]
        a = torch.from_numpy(a).unsqueeze(0).cuda().float()
        b = torch.from_numpy(b).unsqueeze(0).cuda().float()
        with torch.no_grad():
            model(a, b)
        return {k: caps[k] for k in STAGES}, cell_bins(label, q33, q67)

    # fit stats: per (stage, bin): pos/neg sums over the fit half
    fits = {(k, b): {"s1": torch.zeros(D, device="cuda", dtype=torch.float64),
                     "s0": torch.zeros(D, device="cuda", dtype=torch.float64),
                     "q1": torch.zeros(D, device="cuda", dtype=torch.float64),
                     "q0": torch.zeros(D, device="cuda", dtype=torch.float64),
                     "G": torch.zeros(D, D, device="cuda", dtype=torch.float64),
                     "n1": 0, "n0": 0}
            for k in STAGES for b in (1, 2, 3)}
    for i in range(n_fit):
        outs, cb = forward_sample(i)
        for k in STAGES:
            f = outs[k][0].permute(1, 2, 0).reshape(-1, D)
            for b in (1, 2, 3):
                pos = (cb == b).reshape(-1)
                neg = (cb == 0).reshape(-1)
                st = fits[(k, b)]
                fp = f[pos].double()
                fn = f[neg].double()
                st["n1"] += fp.shape[0]
                st["n0"] += fn.shape[0]
                st["s1"] += fp.sum(0)
                st["s0"] += fn.sum(0)
                st["q1"] += (fp ** 2).sum(0)
                st["q0"] += (fn ** 2).sum(0)
                st["G"] += fp.T @ fp + fn.T @ fn

    w_lda = {}
    for (k, b), st in fits.items():
        if st["n1"] == 0 or st["n0"] == 0:
            w_lda[(k, b)] = torch.zeros(D, device="cuda", dtype=torch.float64)
            continue
        mu0 = st["s0"] / st["n0"]
        mu1 = st["s1"] / st["n1"]
        var0 = st["q0"] / st["n0"] - mu0 ** 2
        var1 = st["q1"] / st["n1"] - mu1 ** 2
        den = var0 + var1 + 1e-6
        w_lda[(k, b)] = (mu1 - mu0) / den

    def score_split(lo, hi):
        acc = {(k, b): {"s": [], "y": []} for k in STAGES for b in (1, 2, 3)}
        for i in range(lo, hi):
            outs, cb = forward_sample(i)
            cbf = cb.reshape(-1)
            for k in STAGES:
                f = outs[k][0].permute(1, 2, 0).reshape(-1, D)
                for b in (1, 2, 3):
                    sel = (cbf == b) | (cbf == 0)
                    acc[(k, b)]["s"].append(f[sel].double() @ w_lda[(k, b)])
                    acc[(k, b)]["y"].append(cbf[sel] == b)
        return {(k, b): roc_auc(torch.cat(acc[(k, b)]["s"]), torch.cat(acc[(k, b)]["y"]))
                for k in STAGES for b in (1, 2, 3)}

    auc_ho = score_split(n_fit, n)
    auc_is = score_split(0, n_fit)

    report = {}
    for k in STAGES:
        report[k] = {}
        for b in (1, 2, 3):
            st = fits[(k, b)]
            name = BIN_NAMES[b]
            if st["n1"] == 0 or st["n0"] == 0:
                report[k][name] = {"n_pos": st["n1"], "n_neg": st["n0"],
                                   "note": "no positive/negative cells in fit half"}
                continue
            mu0 = st["s0"] / st["n0"]
            mu1 = st["s1"] / st["n1"]
            var0 = st["q0"] / st["n0"] - mu0 ** 2
            var1 = st["q1"] / st["n1"] - mu1 ** 2
            den = var0 + var1 + 1e-6
            fisher = ((mu1 - mu0) ** 2) / den
            topk = torch.topk(fisher, 5)
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
            report[k][name] = {
                "n_pos": st["n1"], "n_neg": st["n0"],
                "centroid_l2": float((mu1 - mu0).norm().item()),
                "centroid_l2_norm": float(((mu1 - mu0).norm() / (mu1.norm() + mu0.norm() + 1e-12)).item()),
                "fisher_mean": float(fisher.mean().item()),
                "fisher_top5": [{"ch": int(c), "fisher": float(v)}
                                for c, v in zip(topk.indices, topk.values)],
                "eff_rank": round(float(torch.exp(ent).item()), 2),
                "mean_abs_corr": round(float(off.abs().mean().item()), 4),
                "diag_lda_auc_ho": round(auc_ho[(k, b)], 4),
                "diag_lda_auc_is": round(auc_is[(k, b)], 4),
            }

    for h in hooks:
        h.remove()

    # headline comparison: t1 vs fuse1/block1/refine separability per bin
    headline = {}
    for name in BIN_NAMES.values():
        headline[name] = {k: report[k][name].get("diag_lda_auc_ho")
                          for k in STAGES}
    support = {}
    for name in BIN_NAMES.values():
        t1_auc = headline[name]["t1"]
        post_auc = max(headline[name][k] for k in ("fuse1_out", "block1_out", "refine_out")
                       if headline[name][k] is not None)
        support[name] = (t1_auc is not None and post_auc is not None and t1_auc > post_auc)
    print("[HEADLINE] held-out diag-LDA AUROC t1 vs post-fusion stages")
    for name in BIN_NAMES.values():
        print(f"  {name}: t1={headline[name]['t1']} fuse1={headline[name]['fuse1_out']} "
              f"block1={headline[name]['block1_out']} refine={headline[name]['refine_out']} "
              f"-> t1>max(post)={support[name]}")

    out = {"checkpoint": args.checkpoint, "n_pairs": n, "n_fit": n_fit,
           "bin_edges": {"small_max_px": round(q33, 1), "medium_max_px": round(q67, 1),
                         "n_train_components": n_train_comp},
           "headline_t1_gt_postfusion_auc": headline,
           "prefusion_retention_supported": support,
           "stages": report}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "prefusion_retention_profile.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
