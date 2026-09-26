"""Training-only Inner-Boundary Auxiliary Supervision (IBAS) — Run4.

GT boundary target is generated ONLINE from the binary change label:

    B+ = Y - Erode3x3(Y)

i.e. the one-pixel inner ring of each change object (NOT the outside background
ring, and NOT generic image gradients). The valid mask is eroded 3x3 too, so
crop padding (label == 255) can never produce fake boundary targets.

Boundary-head loss (head is deploy-deleted, zero train-params at inference):

    L_b = BCEWithLogits(B+) + Dice(B+)
"""
import torch
import torch.nn.functional as F


def _erode3x3(mask):
    """Exact binary erosion of mask (B,H,W) float {0,1} with a 3x3 ones SE."""
    k = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    s = F.conv2d(mask.unsqueeze(1), k, padding=1)
    return (s >= 8.999).float().squeeze(1)


def build_inner_boundary_target(label):
    """label: (B,H,W) long tensor with values {0,1,255}.

    Returns (boundary, valid):
      boundary: (B,H,W) float {0,1} — inner one-pixel ring of change objects;
      valid:    (B,H,W) float {0,1} — eroded valid region (255-padding excluded).
    """
    y = (label == 1).float()
    valid = (label != 255).float()
    boundary = (y - _erode3x3(y)).clamp(min=0.0)   # B+ = Y - Erode(Y), inner ring only
    valid_b = _erode3x3(valid)                     # exclude 255-neighbourhood
    return boundary * valid_b, valid_b


def boundary_loss(boundary_logits, label, eps=1.0):
    """boundary_logits: (B,1,H,W) raw logits; label: (B,H,W) long {0,1,255}.

    Returns scalar BCE + mean per-sample Dice (averaged over batch)."""
    target, valid = build_inner_boundary_target(label)
    logits = boundary_logits.squeeze(1)            # (B,H,W)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / (valid.sum() + eps)

    prob = torch.sigmoid(logits)
    inter = (prob * target * valid).sum(dim=(1, 2))
    denom = ((prob + target) * valid).sum(dim=(1, 2))
    dice = (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()

    return bce + dice
