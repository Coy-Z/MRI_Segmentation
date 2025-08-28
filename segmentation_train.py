import time
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from tempfile import TemporaryDirectory
from utils.segmentation_util import get_model_instance_unet, sum_IoU, get_transform, get_augment, custom_collate_fn, MRIDataset, Combined_Loss

'''Need to review using regularisation in loss instead of patience-based early stopping.'''

def train(model, device, dims : int, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs : int, patience : int = 15):
    """
    Trains the model and returns the best model based on validation IoU.

    Args:
        model: The segmentation model to train.
        device: Device to use ('cuda' or 'cpu').
        dims (int): Dimensions of the input images.
        criterion: Loss function.
        optimizer: Optimizer.
        dataloaders: Dict of DataLoader objects for 'train' and 'val'.
        scheduler: Learning rate scheduler.
        dataset_sizes: Dict with dataset sizes for 'train' and 'val'.
        num_epochs (int): Number of epochs to train.
        patience (int): Early stopping patience (number of epochs without improvement).

    Returns:
        model: The trained model with the best validation IoU.
    """
    since = time.time()
    model = model.to(device)
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_IoU = 0.0
        epochs_no_improve = 0  # Early stopping counter

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)
            val_IoU = 0.0  # Reset validation IoU for this epoch

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train': # set model mode accordingly
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                acc_IoU = 0.0

                # Iterate over data
                for scan, mask in dataloaders[phase]:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Move data to GPU
                    scan = scan.to(device)      # (D * H * W)
                    mask = mask.to(device).long()  # (D * H * W)
                    scan = scan.unsqueeze(3 - dims)  # 2D (D * 1 * H * W) or 3D (1 * D * H * W)
                    mask = mask.unsqueeze(3 - dims)  # 2D (D * 1 * H * W) or 3D (1 * D * H * W)

                    # No need to code differently for 2D or 3D, since the convolutional layers handle both cases.
                    # Forward pass: Track history if in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        pred_mask_logits = model(scan) # 2D (D * num_classes * H * W) or 3D (num_classes * D * H * W)
                        pred_mask = torch.argmax(pred_mask_logits, dim=3 - dims) # (D * H * W)

                        if dims == 3: # 3D
                            loss = criterion(pred_mask_logits.unsqueeze(0), mask) # Add batch dimension to logits
                        else: # 2D
                            loss = criterion(pred_mask_logits, mask.squeeze(1)) # Remove channel dimensions from mask
                    
                        if phase == 'train':
                            loss.backward()
                            # Add gradient clipping for training stability
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                    
                    # Accumulate Statistics
                    running_loss += loss.item() * scan.size(0) # Ensure the criterion reduction parameter is 'mean'

                    acc_IoU += sum_IoU(pred_mask, mask) # pred_mask broadcasts to mask's shape

                epoch_loss = running_loss / dataset_sizes[phase] # Need to somehow normalize this for batch size, but not necessary.
                epoch_IoU = acc_IoU / dataset_sizes[phase] # The epoch IoU is a mean IoU for the current epoch and phase.
                print(f'{phase} Loss: {epoch_loss:.4f} Mean IoU: {epoch_IoU:.4f}')

                # Deep copy the model
                if phase == 'val' and epoch_IoU > best_IoU:
                    best_IoU = epoch_IoU
                    torch.save(model.state_dict(), best_model_params_path)
                    epochs_no_improve = 0  # Reset counter
                elif phase == 'val':
                    epochs_no_improve += 1  # Increment counter
                    val_IoU = epoch_IoU  # Store validation IoU for scheduler
            
            # Step scheduler once per epoch using validation IoU (outside the phase loop)
            scheduler.step(val_IoU)
            
            # Early stopping check
            if epochs_no_improve >= patience:
                print(f'Early stopping after {epoch + 1} epochs (no improvement for {patience} epochs)')
                break

            # Check for NaN loss
            if epoch_loss != epoch_loss:
                print(f'NaN loss detected at epoch {epoch + 1}. Stopping training.')
                break

            print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')
        print(f'Best validation mean IoU: {best_IoU:4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only = True))
    return model

if __name__ == '__main__':
    # Select dimensions and device
    dims = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
        print(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    # Define transforms.
    # Augments are random changes, which are useful for training but not validation.
    transform = get_transform(data = 'input', dims = dims)
    target_transform = get_transform(data = 'target', dims = dims)
    augment = get_augment(dims = dims)

    # Set up datasets and dataloaders
    data_dir = 'data'
    image_datasets = {x : MRIDataset(root = os.path.join(data_dir, x), phase = x,
                                     dims = dims, transform = transform,
                                     target_transform = target_transform,
                                     augment = augment) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # Set up DataLoader
    num_workers = min(4, os.cpu_count())
    batch_size = 1  # Keep batch_size = 1 due to variable scan sizes and small validation set
    dataloaders = {x: DataLoader(image_datasets[x], batch_size = batch_size, shuffle = True,
                                 num_workers = num_workers, persistent_workers = True if num_workers > 0 else False,
                                 pin_memory = torch.cuda.is_available(),  # Pin memory for faster GPU transfer
                                 collate_fn = custom_collate_fn  # Additional redundancy in case batch_size > 1.
                                 ) for x in ['train', 'val']}
    
    # Initialize model, loss, optimizer, and scheduler
    model = get_model_instance_unet(num_classes = 2, device = device, dims = dims, trained = False)
    criterion = Combined_Loss(device, dims = dims, alpha = 1, ce_weights = (0.3, 0.7))
    
    # Use AdamW with weight decay for L2 regularization
    optimizer = optim.AdamW(model.parameters(), lr = 0.0001, weight_decay = 0.01)

    # Learning rate scheduler reduces learning rate when validation IoU plateaus
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5,
                                                        patience = 10, min_lr = 1e-7)

    print(f"\nDataset sizes: {dataset_sizes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    model = train(model, device, dims, criterion, optimizer, dataloaders, lr_scheduler, dataset_sizes, 
                  num_epochs = 100, patience = 10)

    # Save the model parameters
    torch.save(model.state_dict(), f'{dims}D_model_params.pth')