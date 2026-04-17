#!/usr/bin/env python3
"""
Grad-CAM visualization for v17b (InstanceNorm ResNet50, Sub_1 channel only).

What this does
--------------
1. Loads the v17b config and best checkpoint.
2. Runs the model on the validation set.
3. Auto-picks the MALIGNANT (label==2) sample that the model predicts most
   confidently as malignant — i.e. the clearest positive.
4. Runs 3D Grad-CAM on the final conv block (backbone.layer4) targeting the
   malignant class.
5. Saves a figure with 5 axial slices of the Sub_1 volume (grayscale) with
   the Grad-CAM heatmap overlaid in 'jet'.
6. Also saves a second figure with a larger single middle-slice view that
   is presentation-friendly.

Both PNGs go to the output directory (default:
`/cluster/work/konradj/breast_mri/outputs_v17b/gradcam/`).

Usage on IDUN
-------------
    cd /cluster/home/konradj/ComputerVision
    conda activate breast_mri
    python scripts/gradcam_v17b.py \
        --config configs/idun_v17b.yaml \
        --checkpoint /cluster/work/konradj/breast_mri/outputs_v17b/checkpoints/best.pt

Optional overrides:
    --output_dir /some/other/dir
    --uid <specific-UID-from-split-csv>   # pick a specific sample instead of auto
    --target_class 2                       # 0=normal 1=benign 2=malignant

The script is deliberately self-contained so you can just git pull and run.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (no X server on IDUN login/compute nodes)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make sibling 'src' importable when run from repo root OR from scripts/
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.dataset import BreastMRIDataset, build_sample_list  # noqa: E402
from src.data.transforms import get_val_transforms                # noqa: E402
from src.models.classifier import load_model_checkpoint           # noqa: E402
from src.utils.config import load_config                          # noqa: E402
from src.utils.reproducibility import get_device, seed_everything # noqa: E402


CLASS_NAMES = ["normal", "benign", "malignant"]


# ------------------------------------------------------------------ #
# Grad-CAM implementation (pure PyTorch, works with InstanceNorm)    #
# ------------------------------------------------------------------ #
class GradCAM3D:
    """
    Minimal Grad-CAM for a 3D CNN classifier.

    Hooks the target layer once and stores activations + gradients. For a
    forward+backward pass on a target class, produces a 3D heatmap (D,H,W)
    in [0,1] that has been upsampled to match the input spatial size.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._h1 = target_layer.register_forward_hook(self._save_act)
        # `register_full_backward_hook` gives us gradients wrt this layer's
        # output — exactly what Grad-CAM needs.
        self._h2 = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        # grad_out[0] has shape [B, C, D', H', W']
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)                  # [B, num_classes]
        score = logits[:, target_class].sum()
        score.backward()

        # global-avg-pool gradients over (D,H,W) -> per-channel weights
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # [B,C,1,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True) # [B,1,D',H',W']
        cam = F.relu(cam)

        # upsample to input spatial size
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear",
                            align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        # normalize to [0,1]
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, logits.detach().cpu().numpy()[0]


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@torch.no_grad()
def score_val_samples(model, loader, device):
    """Run model over val set; return list of (uid, label, prob_vector)."""
    model.eval()
    records = []
    for batch in tqdm(loader, desc="Scoring val set"):
        images = batch["image"].to(device)
        logits = model(images).cpu().numpy()
        uids = batch["uid"]
        labels = batch["label"].cpu().numpy()
        for uid, lbl, z in zip(uids, labels, logits):
            records.append({"uid": uid, "label": int(lbl),
                            "probs": softmax(z)})
    return records


def pick_sample(records, target_class: int, requested_uid: str = None):
    """Pick the requested UID, or else the most-confident correct sample."""
    if requested_uid:
        for r in records:
            if r["uid"] == requested_uid:
                return r
        raise SystemExit(f"UID '{requested_uid}' not found in val set.")
    # Correctly labeled + highest prob for the target class
    correct = [r for r in records if r["label"] == target_class]
    if not correct:
        raise SystemExit(
            f"No val samples with label={target_class} "
            f"({CLASS_NAMES[target_class]}). Try a different --target_class."
        )
    correct.sort(key=lambda r: r["probs"][target_class], reverse=True)
    return correct[0]


def find_target_layer(model):
    """
    Return the 'layer4' module of whatever the model's backbone is.
    v17b: BreastClassifier(backbone=MONAI resnet50) -> model.backbone.layer4.
    """
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4
    # Fallback: last Conv3d in the network
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Could not locate a target conv layer for Grad-CAM.")
    return last_conv


# ------------------------------------------------------------------ #
# Plotting                                                            #
# ------------------------------------------------------------------ #
def _slice_axis(vol):
    """Return the axis index of the slice (depth) direction.

    MONAI loads NIfTI as [H, W, D] after EnsureChannelFirstd + Resized with
    spatial_size=(224, 224, 32) → volume is [224, 224, 32]. The slice axis is
    the smallest one. We detect it robustly rather than hardcoding."""
    return int(np.argmin(vol.shape))


def _axial_slice(vol, z, axis):
    """Take the 2D slice at index z along the slice axis."""
    return np.take(vol, indices=z, axis=axis)


def plot_multi_slice(sub1, cam, probs, label, uid, out_path,
                     n_slices=5, alpha=0.45, target_class=2):
    """Grid of N axial slices with heatmap overlay."""
    axis = _slice_axis(sub1)
    D = sub1.shape[axis]
    # Evenly spaced slices, but stay away from edges which are often empty
    z_idx = np.linspace(D * 0.2, D * 0.8, n_slices).astype(int)

    fig, axes = plt.subplots(2, n_slices, figsize=(3.2 * n_slices, 6.8))
    for j, z in enumerate(z_idx):
        img = _axial_slice(sub1, z, axis)
        heat = _axial_slice(cam, z, axis)

        # Row 0: grayscale MRI
        axes[0, j].imshow(img, cmap="gray", vmin=np.percentile(img, 1),
                          vmax=np.percentile(img, 99))
        axes[0, j].set_title(f"slice z={z}", fontsize=10)
        axes[0, j].axis("off")

        # Row 1: overlay
        axes[1, j].imshow(img, cmap="gray", vmin=np.percentile(img, 1),
                          vmax=np.percentile(img, 99))
        axes[1, j].imshow(heat, cmap="jet", alpha=alpha, vmin=0.0, vmax=1.0)
        axes[1, j].axis("off")

    p = probs
    title = (f"Grad-CAM · v17b (InstanceNorm ResNet50) · Sub_1 · "
             f"target class = {CLASS_NAMES[target_class]}\n"
             f"UID {uid}  |  true = {CLASS_NAMES[label]}  |  "
             f"p(normal)={p[0]:.2f}  p(benign)={p[1]:.2f}  "
             f"p(malignant)={p[2]:.2f}")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hero_slice(sub1, cam, probs, label, uid, out_path, alpha=0.45,
                    target_class=2):
    """Single middle-slice figure at larger size — presentation hero shot."""
    axis = _slice_axis(sub1)
    D = sub1.shape[axis]
    # Pick slice with highest Grad-CAM activation (the 'most interesting' one)
    # Move slice axis to front, then mean over (H, W) for each slice.
    cam_moved = np.moveaxis(cam, axis, 0)  # [D, ...]
    per_slice = cam_moved.reshape(D, -1).mean(axis=1)
    z = int(np.argmax(per_slice))

    img = _axial_slice(sub1, z, axis)
    heat = _axial_slice(cam, z, axis)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    axes[0].imshow(img, cmap="gray", vmin=np.percentile(img, 1),
                   vmax=np.percentile(img, 99))
    axes[0].set_title(f"Sub_1 · slice z={z}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(img, cmap="gray", vmin=np.percentile(img, 1),
                   vmax=np.percentile(img, 99))
    im = axes[1].imshow(heat, cmap="jet", alpha=alpha, vmin=0.0, vmax=1.0)
    axes[1].set_title(f"+ Grad-CAM ({CLASS_NAMES[target_class]} class)",
                      fontsize=12)
    axes[1].axis("off")

    p = probs
    title = (f"v17b · UID {uid}  |  true = {CLASS_NAMES[label]}  |  "
             f"p({CLASS_NAMES[target_class]}) = {p[target_class]:.2f}")
    fig.suptitle(title, fontsize=13)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Grad-CAM (normalized)", rotation=270, labelpad=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/idun_v17b.yaml")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Where to write PNGs. Defaults to <paths.output_dir>/gradcam/")
    ap.add_argument("--uid", type=str, default=None,
                    help="Specific UID from the val set. Overrides auto-pick.")
    ap.add_argument("--target_class", type=int, default=2,
                    help="0=normal, 1=benign, 2=malignant (default).")
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="Heatmap overlay alpha.")
    args, overrides = ap.parse_known_args()

    cfg = load_config(args.config, overrides if overrides else None)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    out_dir = Path(args.output_dir) if args.output_dir \
        else Path(cfg.paths.output_dir) / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data: val set with val (non-augmenting) transforms ---
    val_tf = get_val_transforms(
        sequences=cfg.data.sequences,
        spatial_size=cfg.data.spatial_size,
        use_percentile_norm=cfg.augmentation.use_percentile_norm,
        derive_sub2=cfg.augmentation.derive_sub2,
        derive_washout=cfg.augmentation.derive_washout,
        crop_foreground=cfg.augmentation.crop_foreground,
    )
    samples = build_sample_list(
        cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
        cfg.data.sequences, cfg.data.fold, "val",
    )
    ds = BreastMRIDataset(samples, val_tf)

    # Simple batch_size=1 iterator so UID/label line up cleanly
    from torch.utils.data import DataLoader as _DL
    loader = _DL(ds, batch_size=1, shuffle=False,
                 num_workers=min(4, cfg.data.num_workers))

    # --- Model ---
    model = load_model_checkpoint(cfg.model, args.checkpoint, device)
    model.eval()

    # --- Pick the sample ---
    records = score_val_samples(model, loader, device)
    chosen = pick_sample(records, args.target_class, args.uid)
    print(f"\nPicked sample:")
    print(f"  UID:        {chosen['uid']}")
    print(f"  true class: {CLASS_NAMES[chosen['label']]}")
    print(f"  p={chosen['probs'].tolist()}")

    # Re-fetch that sample's tensor
    target_item = None
    for i, s in enumerate(samples):
        if s.uid == chosen["uid"]:
            target_item = ds[i]
            break
    assert target_item is not None
    x = target_item["image"].unsqueeze(0).to(device)  # [1, 1, D, H, W]

    # --- Grad-CAM ---
    target_layer = find_target_layer(model)
    print(f"Target layer for Grad-CAM: {type(target_layer).__name__}")
    cam_engine = GradCAM3D(model, target_layer)
    x.requires_grad_(True)
    cam, logits = cam_engine(x, target_class=args.target_class)
    cam_engine.remove()

    # Volume for plotting (channel 0 = Sub_1)
    sub1 = x.detach()[0, 0].cpu().numpy()          # [D, H, W]
    probs = softmax(logits)

    # --- Save figures ---
    base = f"gradcam_{chosen['uid']}_class{args.target_class}"
    multi_path = out_dir / f"{base}_multi.png"
    hero_path = out_dir / f"{base}_hero.png"

    plot_multi_slice(sub1, cam, probs, chosen["label"], chosen["uid"],
                     multi_path, alpha=args.alpha,
                     target_class=args.target_class)
    plot_hero_slice(sub1, cam, probs, chosen["label"], chosen["uid"],
                    hero_path, alpha=args.alpha,
                    target_class=args.target_class)

    print(f"\nSaved:")
    print(f"  {multi_path}")
    print(f"  {hero_path}")


if __name__ == "__main__":
    main()
