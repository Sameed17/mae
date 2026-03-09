"""
config in once place so  we dont have to copy paste it everywhere . . . sameed
"""

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_CHANNELS = 3
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196

MASK_RATIO = 0.75
NUM_VISIBLE = int(NUM_PATCHES * (1 - MASK_RATIO))  # 49

# encoder parameters
ENC_EMBED_DIM = 768
ENC_DEPTH = 12
ENC_NUM_HEADS = 12

# decoder parameters
DEC_EMBED_DIM = 384
DEC_DEPTH = 12
DEC_NUM_HEADS = 6
