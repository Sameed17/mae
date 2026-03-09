# MAE Image Reconstructor

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open in browser:** `http://localhost:8501`

## Features

- Upload any image (JPG, PNG, GIF, BMP)
- Adjust masking ratio with slider (0-95%)
- View masked and reconstructed images side-by-side
- Get PSNR and SSIM metrics for reconstruction quality
- Download results as PNG

## How It Works

The model:
- Takes an image and masks random patches (default 75%)
- Uses an encoder to process visible patches
- Uses a decoder with learnable mask tokens to predict missing patches
- Reconstruction quality depends on the masking ratio

## Model Info

- **Encoder:** ViT
  - 12-layers each with 12 attention heads
  - Hidden Dimension: 768
- **Decoder:** ViT
  - 12-layers each with 6 attention heads
  - Hidden Dimension: 384
- **Input:** 224×224 RGB images
- **Patches:** 16×16 pixels each (196 patches total)