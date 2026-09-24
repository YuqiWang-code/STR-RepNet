"""Train STR-RepNet (Clean TAR-DCR) for binary remote-sensing change detection.

Frozen VMamba-Tiny Siamese encoder + TAR temporal bridge + DCR decoder.
Logs every epoch as one line (losses + 6 metrics), saves `last.pth` (resume) and
`best_F1=xxx.pth` (test), and appends a final deploy test block at the end.
"""
import os
import sys
import time
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Make `changedetection` and `classification` importable (repo layout: models/...).
_MODELS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from changedetection.configs.config import get_config
from changedetection.datasets.make_data_loader import ChangeDetectionDataset, read_list
from changedetection.utils_func.metrics import Evaluator
from changedetection.utils_func import lovasz_loss as L
from changedetection.models.STRRepNet import STRRepNet


# -----------------------------------------------------------------------------
# Params / FLOPs measurement (eval mode = inference complexity)
# -----------------------------------------------------------------------------
def measure_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_flops(model, size=256):
    """Return (total_flops, n_unsupported) for a (1,3,size,size) bi-temporal pair."""
    from fvcore.nn import flop_count
    from classification.models.vmamba import selective_scan_flop_jit

    supported_ops = {
        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
    }
    model.eval()
    pre = torch.randn(1, 3, size, size).cuda()
    post = torch.randn(1, 3, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post), supported_ops=supported_ops)
    total = sum(counts.values())
    return total, len(unsupported)


def measure_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt_flops(total):
    # fvcore.flop_count already returns counts in giga-flops.
    return f"{total:.4f}"


def fmt_params(n):
    return f"{n / 1e6:.3f}"


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class Trainer(object):
    def __init__(self, args, config, log):
        self.args = args
        self.config = config
        self.log = log

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.model = self._build_model(config)
        self.model = self.model.cuda()

        # param groups: decoder (1e-4) + optionally unfrozen encoder (smaller LR)
        encoder_params, decoder_params = [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                (encoder_params if name.startswith("encoder.") else decoder_params).append(p)
        param_groups = [{"params": decoder_params, "lr": args.learning_rate}]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.learning_rate * args.encoder_lr_ratio})
        self.optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)

        self.train_list = read_list(args.train_list)
        self.test_list = read_list(args.test_list)

        self.start_epoch = 0
        self.best_f1 = -1.0
        self.best_epoch = -1
        resume_path = args.resume
        if resume_path is None:
            auto = os.path.join(args.ckpt_dir, "last.pth")
            if os.path.isfile(auto):
                resume_path = auto
        if resume_path is not None and os.path.isfile(resume_path):
            self._load_resume(resume_path)

        os.makedirs(args.ckpt_dir, exist_ok=True)

    def _build_model(self, config):
        v = config.MODEL.VSSM
        model = STRRepNet(
            pretrained=self.args.pretrained_weight_path,
            rep_mode=self.args.rep_mode,
            use_residual=self.args.use_residual,
            encoder_train=self.args.encoder_train,
            patch_size=v.PATCH_SIZE,
            in_chans=v.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=v.DEPTHS,
            dims=v.EMBED_DIM,
            ssm_d_state=v.SSM_D_STATE,
            ssm_ratio=v.SSM_RATIO,
            ssm_rank_ratio=v.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if v.SSM_DT_RANK == "auto" else int(v.SSM_DT_RANK)),
            ssm_act_layer=v.SSM_ACT_LAYER,
            ssm_conv=v.SSM_CONV,
            ssm_conv_bias=v.SSM_CONV_BIAS,
            ssm_drop_rate=v.SSM_DROP_RATE,
            ssm_init=v.SSM_INIT,
            forward_type=v.SSM_FORWARDTYPE,
            mlp_ratio=v.MLP_RATIO,
            mlp_act_layer=v.MLP_ACT_LAYER,
            mlp_drop_rate=v.MLP_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=v.PATCH_NORM,
            norm_layer=v.NORM_LAYER,
            downsample_version=v.DOWNSAMPLE,
            patchembed_version=v.PATCHEMBED,
            gmlp=v.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        return model

    def _load_resume(self, path):
        self.log(f"[RESUME] loading {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_f1 = ckpt.get("best_f1", -1.0)
        self.best_epoch = ckpt.get("best_epoch", -1)

    def _save(self, path, epoch):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
        }, path)

    def _make_loader(self, list_path, batch_size, shuffle, drop_last):
        dataset = ChangeDetectionDataset(
            self.args.dataset_root, read_list(list_path), self.args.crop_size,
            type=('train' if shuffle else 'test'),
            temporal_swap_prob=(self.args.temporal_swap_prob if shuffle else 0.0),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=self.args.num_workers, drop_last=drop_last, pin_memory=True)

    def _evaluate(self, loader):
        evaluator = Evaluator(num_class=2)
        self.model.eval()
        with torch.no_grad():
            for pre, post, label, _ in loader:
                pre = pre.cuda().float()
                post = post.cuda().float()
                out = self.model(pre, post)
                pred = torch.argmax(out, dim=1).cpu().numpy()  # (B, H, W)
                gt = label.numpy()
                for b in range(pred.shape[0]):
                    evaluator.add_batch(gt[b], pred[b])
        rec = evaluator.Pixel_Recall_Rate()
        pre_ = evaluator.Pixel_Precision_Rate()
        oa = evaluator.Pixel_Accuracy()
        f1 = evaluator.Pixel_F1_score()
        iou = evaluator.Intersection_over_Union()
        kc = evaluator.Kappa_coefficient()
        return rec, pre_, oa, f1, iou, kc

    def train(self):
        train_loader = self._make_loader(self.args.train_list, self.args.batch_size, True, True)
        test_loader = self._make_loader(self.args.test_list, self.args.test_batch_size, False, False)

        self.log(f"[EPOCHS] {self.args.epochs}  [train_iters/epoch] {len(train_loader)}")
        self.log("EPOCH | CE | Lovasz | Total | Recall | Precision | OA | F1 | IoU | Kappa")

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            ce_sum = 0.0
            lovasz_sum = 0.0
            total_sum = 0.0
            n_batches = 0

            for pre, post, label, _ in train_loader:
                pre = pre.cuda().float()
                post = post.cuda().float()
                label = label.cuda().long()

                output = self.model(pre, post)

                ce_loss = F.cross_entropy(output, label, ignore_index=255)
                lovasz = L.lovasz_softmax(F.softmax(output, dim=1), label, ignore=255)
                total_loss = ce_loss + self.args.lovasz_weight * lovasz

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                ce_sum += ce_loss.item()
                lovasz_sum += lovasz.item()
                total_sum += total_loss.item()
                n_batches += 1

            ce_avg = ce_sum / n_batches
            lovasz_avg = lovasz_sum / n_batches
            total_avg = total_sum / n_batches

            # Per-epoch validation on the test set (select best by F1).
            rec, pre_, oa, f1, iou, kc = self._evaluate(test_loader)

            line = (f"Epoch {epoch + 1}/{self.args.epochs} | "
                    f"CE={ce_avg:.4f} | Lovasz={lovasz_avg:.4f} | Total={total_avg:.4f} | "
                    f"Recall={rec:.4f} | Precision={pre_:.4f} | OA={oa:.4f} | "
                    f"F1={f1:.4f} | IoU={iou:.4f} | Kappa={kc:.4f}")
            self.log(line)

            # Save last (resume) + best (test).
            self._save(os.path.join(self.args.ckpt_dir, "last.pth"), epoch)

            if f1 > self.best_f1:
                # remove previous best to keep only the latest best
                prev_best = os.path.join(self.args.ckpt_dir, f"best_F1={self.best_f1:.4f}.pth")
                if self.best_f1 >= 0 and os.path.isfile(prev_best):
                    os.remove(prev_best)
                self.best_f1 = f1
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.ckpt_dir, f"best_F1={self.best_f1:.4f}.pth"))

        self.log(f"[BEST] F1={self.best_f1:.4f} at epoch {self.best_epoch}")

    def test_best(self):
        """Final test with the best checkpoint on the deploy graph."""
        import copy

        best_path = os.path.join(self.args.ckpt_dir, f"best_F1={self.best_f1:.4f}.pth")
        if not os.path.isfile(best_path):
            self.log("[TEST] best checkpoint not found, skipping final test")
            return

        self.model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False))
        self.model = self.model.cuda()

        # disable TF32 to isolate the FP32 re-parameterization fold error
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True

        self.model.eval()
        ref_pre = torch.randn(1, 3, self.args.crop_size, self.args.crop_size).cuda()
        ref_post = torch.randn(1, 3, self.args.crop_size, self.args.crop_size).cuda()
        with torch.no_grad():
            ref_logits = self.model(ref_pre, ref_post)

        deploy_model = copy.deepcopy(self.model)
        deploy_model.switch_to_deploy()
        deploy_model.eval()
        with torch.no_grad():
            dep_logits = deploy_model(ref_pre, ref_post)
        reparam_err = (ref_logits - dep_logits).abs().max().item()

        train_total = measure_params(self.model)
        trainable = measure_trainable_params(self.model)
        deploy_params = measure_params(deploy_model)
        deploy_flops, n_unsup = measure_flops(deploy_model, size=self.args.crop_size)

        test_loader = self._make_loader(self.args.test_list, self.args.test_batch_size, False, False)
        saved_model = self.model
        self.model = deploy_model
        rec, pre_, oa, f1, iou, kc = self._evaluate(test_loader)
        self.model = saved_model

        self.log("=== TEST RESULTS ===")
        self.log("[MODEL] STR-RepNet Clean TAR-DCR")
        self.log(f"[REP-MODE] {self.args.rep_mode}")
        self.log(f"[TOTAL-TRAIN-GRAPH-PARAMS] {fmt_params(train_total)} M")
        self.log(f"[TRAINABLE-PARAMS] {fmt_params(trainable)} M")
        self.log(f"[DEPLOY-PARAMS] {fmt_params(deploy_params)} M")
        self.log(f"[DEPLOY-FLOPS] {fmt_flops(deploy_flops)} G   (unsupported_ops={n_unsup})")
        self.log(f"[REPARAM-MAX-ABS-ERROR] {reparam_err:.3e}")
        self.log(f"Recall={rec:.4f} | Precision={pre_:.4f} | OA={oa:.4f} | F1={f1:.4f} | IoU={iou:.4f} | Kappa={kc:.4f}")
        self.log(f"[BEST-F1] {self.best_f1:.4f} (epoch {self.best_epoch})")
        self.log("=== END TEST RESULTS ===")


# -----------------------------------------------------------------------------
# Config header
# -----------------------------------------------------------------------------
def write_header(args, config, log):
    log("=" * 72)
    log("STR-RepNet Clean TAR-DCR  |  train_scripts/TAR-DCR/Run1")
    log("=" * 72)
    log("[CONFIG]")
    for k, v in vars(args).items():
        log(f"  {k}: {v}")
    log("[MODEL-CONFIG]")
    log(config.dump())
    log("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="HAM-CD baseline training (Run1)")
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--pretrained_weight_path', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lovasz_weight', type=float, default=2.0)
    parser.add_argument('--rep_mode', type=str, default='full', choices=['plain', 'tar', 'dcr', 'full'])
    parser.add_argument('--use_residual', type=int, default=1)
    parser.add_argument('--encoder_train', type=str, default='frozen', choices=['frozen', 'last2', 'full'])
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1)
    parser.add_argument('--temporal_swap_prob', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    config = get_config(args)

    def log(msg):
        print(msg, flush=True)

    write_header(args, config, log)

    trainer = Trainer(args, config, log)

    log("[TOTAL-TRAIN-GRAPH-PARAMS] " + fmt_params(measure_params(trainer.model)) + " M")
    log("[TRAINABLE-PARAMS] " + fmt_params(measure_trainable_params(trainer.model)) + " M")
    flops, n_unsup = measure_flops(trainer.model, size=args.crop_size)
    log(f"[FLOPS]  {fmt_flops(flops)} G   (input 2x3x{args.crop_size}x{args.crop_size}, unsupported_ops={n_unsup})")
    log("=" * 72)

    trainer.train()
    trainer.test_best()


if __name__ == "__main__":
    main()
