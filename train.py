import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import TinyImageNet
from models import MAE, mae_loss, image_to_patch_pixels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fraction = 0.1  # training on 1% of train set
    full_train = TinyImageNet("tiny-imagenet-200", split="train")
    full_val = TinyImageNet("tiny-imagenet-200", split="val")
    n_train = len(full_train)
    n_val = len(full_val)
    train_set = Subset(full_train, torch.randperm(n_train)[:max(1, int(n_train * fraction))].tolist())
    val_set = Subset(full_val, torch.randperm(n_val)[:max(1, int(n_val * fraction))].tolist())
    train_loader = DataLoader(train_set, batch_size=8)
    val_loader = DataLoader(val_set, batch_size=8)
    print(f"Train: {len(train_set)} ({100*fraction:.0f}% of {n_train})  Val: {len(val_set)} ({100*fraction:.0f}% of {n_val})")

    model = MAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    state = torch.load("model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    model.train()
    for epoch in range(1, 20):
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            x = batch.to(device)

            pred_masked, visible_indices, mask_indices = model(x)
            target_patches = image_to_patch_pixels(x)
            loss = mae_loss(pred_masked, target_patches, mask_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                x = batch.to(device)
                pred_masked, _, mask_indices = model(x)
                target_patches = image_to_patch_pixels(x)
                val_loss += mae_loss(pred_masked, target_patches, mask_indices).item()
                val_batches += 1
        model.train()
        val_loss = val_loss / val_batches

        print(f"Epoch {epoch}  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")
        torch.save(model.state_dict(), "model.pt")
        
    print("Training finished.")

if __name__ == "__main__":
    main()