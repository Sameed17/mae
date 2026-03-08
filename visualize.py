"""
MAE visualization: masked input, model reconstruction, and ground truth.
Run from project root:  python visualize.py

Displays at least 5 qualitative reconstruction examples in a 3-panel grid:
  • Masked Input (75% patches removed)
  • Model Reconstruction (visible + predicted masked)
  • Original Ground Truth
"""

import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

from config import IMAGE_SIZE, NUM_PATCHES
from dataset import TinyImageNet
from models import MAE, image_to_patch_pixels, patch_pixels_to_image


def build_masked_image(x, mask_indices, fill=0.5):
    """
    Replace masked patches with a constant (e.g. gray).
    x: (B, 3, H, W), mask_indices: (B, num_masked)
    Returns: (B, 3, H, W) with masked patches set to fill.
    """
    B, C, H, W = x.shape
    target_patches = image_to_patch_pixels(x)  # (B, 196, 768)
    # Replace masked positions with gray patch
    patch_dim = target_patches.shape[-1]
    gray_patch = torch.full((1, 1, patch_dim), fill, device=x.device, dtype=x.dtype)
    gray_patch = gray_patch.expand(B, mask_indices.shape[1], -1)
    mask_exp = mask_indices.unsqueeze(-1).expand(-1, -1, patch_dim)
    target_patches = target_patches.clone()
    target_patches.scatter_(1, mask_exp, gray_patch)
    return patch_pixels_to_image(target_patches)


def build_reconstruction_image(x, pred_masked, visible_indices, mask_indices):
    """
    Full image: visible patches from original, masked patches from model prediction.
    Model predicts normalized patch pixels; we denormalize using target mean/var before
    assembling the image (same normalization as in mae_loss).
    x: (B, 3, H, W), pred_masked: (B, num_masked, patch_pixels) in normalized space
    Returns: (B, 3, H, W).
    """
    target_patches = image_to_patch_pixels(x)  # (B, 196, patch_dim)
    patch_dim = target_patches.shape[-1]
    mask_exp = mask_indices.unsqueeze(-1).expand(-1, -1, patch_dim)
    target_masked = torch.gather(target_patches, 1, mask_exp)

    # Denormalize: model predicts (target - mean) / std
    mean = target_masked.mean(dim=-1, keepdim=True)
    var = target_masked.var(dim=-1, keepdim=True)
    pred_pixels = pred_masked * torch.sqrt(var + 1e-6) + mean

    target_patches = target_patches.clone()
    target_patches.scatter_(1, mask_exp, pred_pixels)
    return patch_pixels_to_image(target_patches)


def visualize_reconstructions(
    model,
    dataloader,
    device,
    num_examples=5,
    save_path="mae_reconstructions.png",
):
    """
    Run model on batches, collect 4-panel visuals for at least num_examples samples.
    Uses a fixed mask (same seed) for reproducibility.
    """
    model.eval()
    B = 1
    # Fixed mask for all samples (we'll use the same mask pattern per batch position)
    rand = torch.rand(B, NUM_PATCHES, device=device)
    ids = rand.argsort(dim=1)
    num_visible = model.num_visible
    visible_indices = ids[:, :num_visible]
    mask_indices = ids[:, num_visible:]

    collected = []  # list of (masked_im, recon_im, full_im, gt_im) per sample

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            # Expand fixed mask to batch size (same mask for all in batch for simplicity)
            vis = visible_indices.expand(x.shape[0], -1)
            msk = mask_indices.expand(x.shape[0], -1)

            pred_masked, v, m = model(x, visible_indices=vis, mask_indices=msk)
            masked_im = build_masked_image(x, m, fill=0.5)
            recon_im = build_reconstruction_image(x, pred_masked, v, m)

            for i in range(x.shape[0]):
                collected.append((
                    masked_im[i].cpu(),
                    recon_im[i].cpu(),
                    x[i].cpu(),
                ))
                if len(collected) >= num_examples:
                    break
            if len(collected) >= num_examples:
                break

    n = len(collected)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    titles = ["Masked Input (75% removed)", "Model Reconstruction", "Original Ground Truth"]
    for i in range(3):
        axes[0, i].set_title(titles[i], fontsize=11)
    for row, (masked, recon, gt) in enumerate(collected):
        for col, img in enumerate([masked, recon, gt]):
            ax = axes[row, col]
            # (C, H, W) -> (H, W, C), clip to [0, 1]
            arr = img.permute(1, 2, 0).numpy()
            arr = np.clip(arr, 0.0, 1.0)
            ax.imshow(arr)
            ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {n} qualitative examples to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data: small subset for visualization
    full_val = TinyImageNet("tiny-imagenet-200", split="val")
    indices = torch.randperm(len(full_val))[: max(5, 16)].tolist()
    val_set = Subset(full_val, indices)
    val_loader = DataLoader(val_set, batch_size=4)

    model = MAE().to(device)
    ckpt = "mae_tiny_imagenet.pt"
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt}")
    except FileNotFoundError:
        print(f"No checkpoint found at {ckpt}; using randomly initialized model for demo.")

    visualize_reconstructions(
        model,
        val_loader,
        device,
        num_examples=5,
        save_path="mae_reconstructions.png",
    )


if __name__ == "__main__":
    main()
