"""LEVIR zero-cost error profile diagnostics (Run6, read-only, no training).

1. Component-size stratified FN analysis: connected-component areas from TRAIN GT
   define the size bins (quantiles, frozen), then per-bin pixel recall and
   completely-missed-component rate are measured on TEST with a given checkpoint.
2. Test-scale sensitivity: single checkpoint evaluated at 192/256/320/384 inputs
   (resize -> model -> resize logits back -> argmax -> metrics). Diagnostic only;
   never used as the final method.

Usage (on server):
  python analyse/levir_error_profile.py \
    --cfg <vssm yaml> --pretrained_weight_path <vssm pth> \
    --checkpoint <best_F1 pth> --dataset_root /share_datasets/CD/LEVIR-CD-256 \
    --train_list .../list/train.txt --test_list .../list/test.txt \
    --output_dir /home/yqwang/outputs/STR-RepNet/diagnostics/Run6_LEVIR \
    --test_scales 192 256 320 384 --gpu 0
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
from changedetection.utils_func.metrics import Evaluator

try:
    from scipy import ndimage as ndi
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def build_model(args, config):
    v = config.MODEL.VSSM
    return STRRepNet(
        pretrained=None, rep_mode="full", use_residual=True, encoder_train="last2",
        use_botr=False, use_nscr=False,
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


def load_model(args, config, device):
    model = build_model(args, config).to(device)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.switch_to_deploy()
    model.eval()
    return model


def evaluate(model, dataset, scale=None, max_n=None, device="cuda"):
    ev = Evaluator(num_class=2)
    n = len(dataset)
    idx = np.arange(n)
    if max_n is not None and n > max_n:
        idx = np.random.RandomState(0).choice(n, max_n, replace=False)
    with torch.no_grad():
        for i in idx:
            a, b, label, _ = dataset[i]
            a = torch.from_numpy(a).unsqueeze(0).to(device).float()
            b = torch.from_numpy(b).unsqueeze(0).to(device).float()
            if scale is not None and scale != 256:
                a = F.interpolate(a, size=(scale, scale), mode="bilinear", align_corners=False)
                b = F.interpolate(b, size=(scale, scale), mode="bilinear", align_corners=False)
            out = model(a, b)
            if scale is not None and scale != 256:
                out = F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)
            pred = torch.argmax(out, dim=1)[0].cpu().numpy()
            ev.add_batch(label, pred)
    rec = ev.Pixel_Recall_Rate()
    pre = ev.Pixel_Precision_Rate()
    f1 = ev.Pixel_F1_score()
    iou = ev.Intersection_over_Union()
    return {"Recall": rec, "Precision": pre, "F1": f1, "IoU": iou}


def component_profile(model, dataset_root, train_list, test_list, output_dir):
    """Component-size stratified FN analysis using TRAIN-GT quantile bins."""
    out = {"has_scipy": HAS_SCIPY}
    if not HAS_SCIPY:
        out["error"] = "scipy not available; skip component profile"
        return out
    # train GT component areas (sample for speed)
    train_ds = ChangeDetectionDataset(dataset_root, read_list(train_list), 256, type="test")
    areas = []
    n_tr = len(train_ds)
    idx = np.random.RandomState(0).choice(n_tr, min(600, n_tr), replace=False)
    for i in idx:
        _, _, lab, _ = train_ds[i]
        mask = (lab == 1).astype(np.uint8)
        if mask.sum() == 0:
            continue
        lbl, ncomp = ndi.label(mask, structure=np.ones((3, 3)))
        for c in range(1, ncomp + 1):
            areas.append(int((lbl == c).sum()))
    areas = np.array(areas, dtype=np.float64)
    q33, q67 = np.quantile(areas, [0.33, 0.67]) if areas.size else (64.0, 256.0)
    bins = [("small", 0, q33), ("medium", q33, q67), ("large", q67, 1e12)]
    out["n_train_components"] = int(areas.size)
    out["bin_edges"] = {"small_max": float(q33), "medium_max": float(q67)}

    # test profile
    test_ds = ChangeDetectionDataset(dataset_root, read_list(test_list), 256, type="test")
    stats = {b[0]: {"gt_px": 0, "tp_px": 0, "fn_px": 0, "n_comp": 0, "missed": 0} for b in bins}
    with torch.no_grad():
        for i in range(len(test_ds)):
            a, b, label, _ = test_ds[i]
            a = torch.from_numpy(a).unsqueeze(0).cuda().float()
            b = torch.from_numpy(b).unsqueeze(0).cuda().float()
            pred = torch.argmax(model(a, b), dim=1)[0].cpu().numpy()
            gt = label
            mask = (gt == 1).astype(np.uint8)
            if mask.sum() == 0:
                continue
            lbl, ncomp = ndi.label(mask, structure=np.ones((3, 3)))
            predmask = (pred == 1).astype(np.uint8)
            for c in range(1, ncomp + 1):
                comp = (lbl == c)
                area = int(comp.sum())
                bname = "small" if area <= q33 else ("medium" if area <= q67 else "large")
                s = stats[bname]
                s["n_comp"] += 1
                s["gt_px"] += area
                tp = int((comp & predmask).sum())
                s["tp_px"] += tp
                s["fn_px"] += area - tp
                if tp == 0:
                    s["missed"] += 1
    rows = []
    for b in bins:
        s = stats[b[0]]
        rec = s["tp_px"] / max(s["gt_px"], 1)
        rows.append({
            "bin": b[0], "gt_components": s["n_comp"], "gt_pixels": s["gt_px"],
            "tp_pixels": s["tp_px"], "fn_pixels": s["fn_px"],
            "pixel_recall": round(rec, 4),
            "completely_missed": s["missed"],
            "missed_rate": round(s["missed"] / max(s["n_comp"], 1), 4),
        })
    out["rows"] = rows
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "component_metrics.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, default="")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--train_list", type=str, required=True)
    ap.add_argument("--test_list", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--test_scales", type=int, nargs="+", default=[256])
    ap.add_argument("--component_max_n", type=int, default=800)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    # attributes required by get_config()
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
    os.makedirs(args.output_dir, exist_ok=True)

    print("building + loading", args.checkpoint)
    model = load_model(args, config, "cuda")
    print("model loaded (deploy graph)")

    # 1) scale sensitivity (256 always; on a fixed subset for the other scales)
    test_ds = ChangeDetectionDataset(args.dataset_root, read_list(args.test_list), 256, type="test")
    scale_rows = []
    for s in args.test_scales:
        maxn = None if s == 256 else 200
        m = evaluate(model, test_ds, scale=s, max_n=maxn, device="cuda")
        m["scale"] = s
        scale_rows.append(m)
        print(f"scale {s}: {m}")
    with open(os.path.join(args.output_dir, "scale_sensitivity.json"), "w") as f:
        json.dump(scale_rows, f, indent=2)

    # 2) component-size FN profile
    prof = component_profile(model, args.dataset_root, args.train_list, args.test_list, args.output_dir)
    print(json.dumps(prof, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
