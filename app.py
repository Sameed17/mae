import streamlit as st
import torch
import numpy as np
from PIL import Image
import io

from config import IMAGE_SIZE, NUM_PATCHES
from models import MAE, image_to_patch_pixels, patch_pixels_to_image
from visualize import build_masked_image, build_reconstruction_image, compute_psnr_ssim


@st.cache_resource
def load_model(device):
    """Load the pre-trained MAE model."""
    model = MAE().to(device)
    state = torch.load("model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor."""
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    tensor = torch.clamp(tensor, 0.0, 1.0)
    img_array = (tensor.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(img_array)


def run_inference(model, image_tensor, mask_ratio, device):
    """Run MAE inference on an image."""
    model.eval()
    
    # Create mask
    batch_size = 1
    rand = torch.rand(batch_size, NUM_PATCHES, device=device)
    ids = rand.argsort(dim=1)
    num_visible = int(NUM_PATCHES * (1 - mask_ratio))
    visible_indices = ids[:, :num_visible]
    mask_indices = ids[:, num_visible:]
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred_masked, _, _ = model(image_tensor, visible_indices=visible_indices, mask_indices=mask_indices)
        
        # Create visualizations
        masked_img = build_masked_image(image_tensor, mask_indices, fill=0.5)
        recon_img = build_reconstruction_image(
            image_tensor, pred_masked, visible_indices, mask_indices
        )
        
        # Compute metrics
        psnr, ssim = compute_psnr_ssim(
            recon_img.squeeze(0).cpu(),
            image_tensor.squeeze(0).cpu(),
            data_range=1.0
        )
    
    # Convert to PIL Images
    masked_pil = tensor_to_image(masked_img.squeeze(0).cpu())
    recon_pil = tensor_to_image(recon_img.squeeze(0).cpu())
    
    return masked_pil, recon_pil, psnr, ssim


def main():
    st.set_page_config(page_title="MAE", layout="wide")
    
    st.title("Masked AutoEncoder")
    st.markdown("Upload an image and watch the model reconstruct it from partial information!")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sidebar
    mask_ratio = st.sidebar.slider("Masking Ratio", 0.0, 0.95, 0.75, 0.05)
    st.sidebar.metric("Visible Patches", f"{int(NUM_PATCHES * (1 - mask_ratio))}/{NUM_PATCHES}")
    
    # Load model
    try:
        model = load_model(device)
        st.sidebar.success("Model loaded")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "gif", "bmp"])
    
    if uploaded_file:
        original_image = Image.open(uploaded_file)
        original_tensor = image_to_tensor(original_image)
        
        with st.spinner("Running inference..."):
            masked_pil, recon_pil, psnr, ssim = run_inference(
                model, original_tensor, mask_ratio, device
            )
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        col1.image(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)), caption="Original")
        col2.image(masked_pil, caption=f"Masked ({mask_ratio:.0%})")
        col3.image(recon_pil, caption="Reconstructed")
        
        col4.metric("PSNR", f"{psnr:.2f} dB")
        col4.metric("SSIM", f"{ssim:.4f}")
        
        # Downloads
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        buf_masked = io.BytesIO()
        masked_pil.save(buf_masked, format="PNG")
        col1.download_button("Download Masked", buf_masked.getvalue(), "masked.png", "image/png")
        
        buf_recon = io.BytesIO()
        recon_pil.save(buf_recon, format="PNG")
        col2.download_button("Download Reconstruction", buf_recon.getvalue(), "reconstruction.png", "image/png")
        
        # Comparison image
        comparison = Image.new('RGB', (IMAGE_SIZE * 2, IMAGE_SIZE))
        comparison.paste(masked_pil, (0, 0))
        comparison.paste(recon_pil, (IMAGE_SIZE, 0))
        
        buf_comparison = io.BytesIO()
        comparison.save(buf_comparison, format="PNG")
        col3.download_button("Download Comparison", buf_comparison.getvalue(), "comparison.png", "image/png")
    else:
        st.info("Upload an image to get started!")


if __name__ == "__main__":
    main()
