import torch
import torch.nn as nn

from config import (
    IMAGE_SIZE, PATCH_SIZE, NUM_CHANNELS, NUM_PATCHES, NUM_VISIBLE,
    ENC_EMBED_DIM, ENC_DEPTH, ENC_NUM_HEADS,
    DEC_EMBED_DIM, DEC_DEPTH, DEC_NUM_HEADS,
)

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.num_patches = NUM_PATCHES

        # Each patch: 16*16*3 = 768 values -> one vector of size embed_dim (which os 768)
        patch_dim = PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        B, C, H, W = x.shape
        p = self.patch_size
        assert H == W == IMAGE_SIZE

        # reshape into patches: (B, 3, 14, 16, 14, 16) -> (B, 14*14, 16*16*3)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, 14, 14, 16, 16, 3)
        x = x.reshape(B, self.num_patches, -1)  # (B, 196, 768)

        return self.proj(x)  # (B, 196, embed_dim)

# vit base
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(ENC_EMBED_DIM)
        self.num_patches = NUM_PATCHES

        # learnable positional embeddings (one per patch position)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, ENC_EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # stack of transformer layers (built-in by pytorch, pls dont touch these muzammil)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                ENC_EMBED_DIM,
                ENC_NUM_HEADS,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(ENC_DEPTH)
        ])
        self.norm = nn.LayerNorm(ENC_EMBED_DIM)

    def forward(self, x, visible_indices):
        """
        x: (B, 3, 224, 224) — full image
        visible_indices: (B, num_visible) — which patch indices are visible (no mask tokens)
        Returns: (B, num_visible, ENC_EMBED_DIM) — latent only for visible patches
        """
        # patch embed full image so we can select visible patches
        tokens = self.patch_embed(x)  # (B, 196, 768)

        # add positional embedding (same for all samples; we index by position)
        tokens = tokens + self.pos_embed  # (B, 196, 768)

        # Keep only visible tokens, encoder never sees masked patches
        # visible_indices: (B, 49) -> gather tokens at those positions
        B, N, D = tokens.shape
        visible_indices = visible_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, 49, 768)
        tokens = torch.gather(tokens, 1, visible_indices)  # (B, 49, 768)

        # Transformer layers
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = NUM_PATCHES

        # encoder outputs are 768-dim; we project to decoder dim 384
        self.enc_to_dec = nn.Linear(ENC_EMBED_DIM, DEC_EMBED_DIM)

        # learnable mask tokens (one vector per masked position, same for all positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DEC_EMBED_DIM))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # positional embeddings in decoder (so it knows patch order for reconstruction)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, DEC_EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                DEC_EMBED_DIM,
                DEC_NUM_HEADS,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(DEC_DEPTH)
        ])
        self.norm = nn.LayerNorm(DEC_EMBED_DIM)

        # predict patch pixels: each decoder output = 16*16*3 values
        self.head = nn.Linear(DEC_EMBED_DIM, PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS)

    def forward(self, latent, visible_indices, mask_indices):
        """
        latent: (B, num_visible, ENC_EMBED_DIM) from encoder
        visible_indices: (B, num_visible): positions of visible patches
        mask_indices: (B, num_masked): positions we must reconstruct
        Returns: (B, num_masked, patch_pixels): predicted pixel values for masked patches only
        """
        B, num_visible, _ = latent.shape
        num_masked = mask_indices.shape[1]

        # project encoder latent to decoder dimension
        latent = self.enc_to_dec(latent)  # (B, 49, 384)

        # build full sequence in original patch order: put latent at visible positions,
        # mask_token at masked positions
        tokens = self.mask_token.expand(B, self.num_patches, -1).clone()  # (B, 196, 384)
        visible_indices_exp = visible_indices.unsqueeze(-1).expand(-1, -1, DEC_EMBED_DIM)
        tokens.scatter_(1, visible_indices_exp, latent)

        # add positional embeddings
        tokens = tokens + self.pos_embed

        # transformer layers
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # predict pixels for all patches, then keep only masked positions for loss
        pred_all = self.head(tokens)  # (B, 196, patch_pixels)
        mask_indices_exp = mask_indices.unsqueeze(-1).expand(-1, -1, pred_all.shape[-1])
        pred_masked = torch.gather(pred_all, 1, mask_indices_exp)  # (B, num_masked, patch_pixels)
        return pred_masked

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.num_patches = NUM_PATCHES
        self.num_visible = NUM_VISIBLE
        self.mask_ratio = 1 - (NUM_VISIBLE / NUM_PATCHES)

    def _random_mask(self, B, device):
        """
        For each sample, shuffle patch indices and split into visible / masked.
        Returns:
            visible_indices: (B, num_visible)
            mask_indices: (B, num_masked)
        """
        num_masked = self.num_patches - self.num_visible
        # Random permutation per sample
        rand = torch.rand(B, self.num_patches, device=device)
        ids = rand.argsort(dim=1)  # (B, 196)
        visible_indices = ids[:, :self.num_visible]      # (B, 49): first 49 will be visible
        mask_indices = ids[:, self.num_visible:]         # (B, 147): last 147 will be hiddel
        return visible_indices, mask_indices

    def forward(self, x, visible_indices=None, mask_indices=None):
        """
        x: (B, 3, 224, 224)
        """
        B = x.shape[0]
        device = x.device
        if visible_indices is None or mask_indices is None:
            visible_indices, mask_indices = self._random_mask(B, device)

        # only visible patches -> latent rep
        latent = self.encoder(x, visible_indices)  # (B, 49, 768)

        # latent rep + mask tokens -> predict masked patch pixels
        pred_masked = self.decoder(latent, visible_indices, mask_indices)  # (B, 147, 768)
        return pred_masked, visible_indices, mask_indices

# helper for loss
def image_to_patch_pixels(x):
    """
    turns image into patch pixels (no projection). Same grid as PatchEmbedding.
    x: (B, 3, 224, 224) -> out: (B, 196, 16*16*3)
    """
    B, C, H, W = x.shape
    p = PATCH_SIZE
    x = x.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, NUM_PATCHES, -1)


def patch_pixels_to_image(patches):
    """
    inverse of image_to_patch_pixels. Reassemble patch pixels into image.
    patches: (B, 196, 16*16*3) -> out: (B, 3, 224, 224)
    """
    B, N, D = patches.shape
    p = PATCH_SIZE
    side = (N ** 0.5)
    assert side == int(side), "NUM_PATCHES must be a perfect square"
    side = int(side)
    # (B, 196, 768) -> (B, 14, 14, 16, 16, 3)
    patches = patches.reshape(B, side, side, p, p, NUM_CHANNELS)
    # (B, 14, 14, 16, 16, 3) -> (B, 3, 14, 16, 14, 16)
    x = patches.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

def mae_loss(model_out, target_patches, mask_indices):
    """
    model_out: (B, num_masked, patch_pixels): model prediction for masked patches (normalized space)
    target_patches: (B, 196, patch_pixels): ground-truth patch pixels from image
    mask_indices: (B, num_masked): which positions were masked

    patch normalization (per-patch mean/var) so the model learns structure rather than average brightness, improving reconstruction quality (mae paper says so).
    """
    B, num_masked, patch_dim = model_out.shape
    mask_exp = mask_indices.unsqueeze(-1).expand(-1, -1, patch_dim)
    target_masked = torch.gather(target_patches, 1, mask_exp)  # (B, num_masked, patch_dim)

    # per-patch normalization: equalize brightness/contrast so MSE focuses on structure
    mean = target_masked.mean(dim=-1, keepdim=True)
    var = target_masked.var(dim=-1, keepdim=True)
    target_masked = (target_masked - mean) / torch.sqrt(var + 1e-6)

    return nn.functional.mse_loss(model_out, target_masked)