from classification.models.vmamba import VSSM, LayerNorm2d
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Run9 BiFTR: Bi-sided Frozen-to-Trainable Transition Reparameterization.
# Wraps the stage2->stage3 downsample Conv (192->384, stride 2) — the frozen/
# trainable boundary under encoder_train="last2".
#
#   train : y = (I + d_out) ( W ((I + d_in) x) + b )      d_in/d_out zero-init 1x1
#   deploy: single original Conv2d with W_eq = A W B, b_eq = A b  (FP64, one cast)
#
# NOTE on kernel size: this repo's config uses downsample_version="v3", so the
# actual operator is Conv2d(192->384, k=3, s=2, p=1) (the design doc assumed v2
# k=2; the AWB fold is kernel-size-agnostic, and the backbone operator is kept
# untouched). The assert enforces the frozen->trainable boundary identity
# (192->384, stride 2) and records the kernel size.
#
# The base Conv stays frozen; only d_in/d_out are trainable (re-enabled AFTER
# the encoder freeze setup — see STRRepNet).
# -----------------------------------------------------------------------------
class BiFTRTransition(nn.Module):
    def __init__(self, conv, mode="bi"):
        super().__init__()
        assert isinstance(conv, nn.Conv2d), "BiFTR wraps the downsample nn.Conv2d"
        assert conv.in_channels == 192 and conv.out_channels == 384, \
            f"BiFTR stage2_to_stage3 must be 192->384, got {conv.in_channels}->{conv.out_channels}"
        assert conv.stride == (2, 2), f"BiFTR requires stride 2, got s={conv.stride}"
        assert mode in ("post", "bi")
        self.mode = mode
        self.conv = conv
        self.d_out = nn.Conv2d(384, 384, 1, bias=False)
        nn.init.zeros_(self.d_out.weight)
        if mode == "bi":
            self.d_in = nn.Conv2d(192, 192, 1, bias=False)
            nn.init.zeros_(self.d_in.weight)
        self.deploy = False

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        if self.mode == "bi":
            x = x + self.d_in(x)
        z = self.conv(x)
        z = z + self.d_out(z)
        return z

    def get_equivalent_kernel_bias(self):
        """FP64: W_eq = (I+d_out) W (I+d_in), b_eq = (I+d_out) b."""
        W = self.conv.weight.detach().double()                       # (384,192,2,2)
        b = (self.conv.bias.detach().double() if self.conv.bias is not None
             else torch.zeros(384, dtype=torch.float64, device=W.device))
        A = torch.eye(384, dtype=torch.float64, device=W.device) \
            + self.d_out.weight[:, :, 0, 0].detach().double()
        if self.mode == "bi":
            B = torch.eye(192, dtype=torch.float64, device=W.device) \
                + self.d_in.weight[:, :, 0, 0].detach().double()
        else:
            B = torch.eye(192, dtype=torch.float64, device=W.device)
        W_pre = torch.einsum("mnxy,ni->mixy", W, B)
        W_eq = torch.einsum("om,mixy->oixy", A, W_pre)
        b_eq = A @ b
        return W_eq, b_eq

    def branch_stats(self):
        """Mechanism record only (not checkpoint selection)."""
        W = self.conv.weight.detach().double()
        d_out_m = self.d_out.weight[:, :, 0, 0].detach().double()
        d_in_m = (self.d_in.weight[:, :, 0, 0].detach().double()
                  if hasattr(self, "d_in") else torch.zeros_like(d_out_m)[:192, :192])
        W_eq, _ = self.get_equivalent_kernel_bias()
        upd = W_eq - W
        upd_norm = (upd.norm() / W.norm()).item() if W.norm() > 0 else 0.0
        # linear contributions: d_out W + W d_in
        W_lin = (torch.einsum("om,mnxy->onxy", d_out_m, W)
                 + torch.einsum("mnxy,ni->mixy", W, d_in_m))
        cross = upd - W_lin
        ratio = (cross.norm() / upd.norm()).item() if upd.norm() > 0 else 0.0
        return {"d_in": d_in_m.norm().item(), "d_out": d_out_m.norm().item(),
                "eff_update": upd_norm, "cross_ratio": ratio}

    def switch_to_deploy(self):
        if self.deploy:
            return self
        W_eq, b_eq = self.get_equivalent_kernel_bias()
        conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels,
                         self.conv.kernel_size, self.conv.stride, self.conv.padding,
                         groups=self.conv.groups, bias=(self.conv.bias is not None))
        conv.weight.data = W_eq.float()
        if conv.bias is not None:
            conv.bias.data = b_eq.float()
        self.conv = conv
        self.deploy = True
        for name in ("d_in", "d_out"):
            if hasattr(self, name):
                delattr(self, name)
        return self


def install_biftr_transition(backbone, mode="bi", transition="stage2_to_stage3"):
    """Wrap layers[1].downsample's Conv with BiFTRTransition (assert-located)."""
    if transition != "stage2_to_stage3":
        raise ValueError(f"unsupported BiFTR transition: {transition}")
    layer = backbone.layers[1]
    ds = layer.downsample
    conv = ds[1]
    assert isinstance(conv, nn.Conv2d), "downsample[1] must be the nn.Conv2d"
    assert conv.in_channels == 192 and conv.out_channels == 384, \
        f"downsample conv must be 192->384, got {conv.in_channels}->{conv.out_channels}"
    assert conv.stride == (2, 2), f"downsample conv must be stride 2, got s={conv.stride}"
    print(f"[BiFTR] wrapping stage2->3 downsample Conv {conv.in_channels}->{conv.out_channels}, "
          f"k={conv.kernel_size}, s={conv.stride}, p={conv.padding} (config v3)")
    wrapper = BiFTRTransition(conv, mode=mode)
    ds[1] = wrapper
    return wrapper


class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        # norm_layer='ln'
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs

