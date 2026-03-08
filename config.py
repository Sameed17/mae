"""
Minimal config: only what the model needs.
Training settings are in train.py.
"""

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_CHANNELS = 3
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196

MASK_RATIO = 0.75
NUM_VISIBLE = int(NUM_PATCHES * (1 - MASK_RATIO))  # 49

# Encoder: ViT-Base (B/16)
ENC_EMBED_DIM = 768
ENC_DEPTH = 12
ENC_NUM_HEADS = 12

# Decoder: ViT-Small (S/16)
DEC_EMBED_DIM = 384
DEC_DEPTH = 12
DEC_NUM_HEADS = 6
