import time
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from tempfile import TemporaryDirectory
from utils.fcn_resnet101_util import clip_and_scale, get_model_instance_segmentation, sum_IoU, get_transform, custom_collate_fn, MRIDataset, Combined_Loss

'''Need to review using regularisation in loss instead of patience-based early stopping.'''

def train(model, device, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs, patience=15):
    """
    Trains the model and returns the best model based on validation IoU.

    Args:
        model: The segmentation model to train.
        device: Device to use ('cuda' or 'cpu').
        criterion: Loss function.
        optimizer: Optimizer.
        dataloaders: Dict of DataLoader objects for 'train' and 'val'.
        scheduler: Learning rate scheduler.
        dataset_sizes: Dict with dataset sizes for 'train' and 'val'.
        num_epochs: Number of epochs to train.
        patience: Early stopping patience (number of epochs without improvement).

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
                for scan, mask3d in dataloaders[phase]:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Move data to GPU
                    scan = scan.to(device)      # (D * 3 * H * W)
                    mask3d = mask3d.to(device).long()  # (D * H * W)

                    # Process the 3D volume as a batch of 2D slices
                    # The scans are already in the right format -> this allows us to process all slices in parallel

                    # Forward pass: Track history if in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(scan)
                        pred_mask3d_logits = outputs['out']
                        pred_mask3d = torch.argmax(outputs['out'], dim = 1)
                        loss = criterion(pred_mask3d_logits, mask3d)

                        if phase == 'train':
                            loss.backward()
                            # Add gradient clipping for training stability
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                    
                    # Accumulate Statistics
                    running_loss += loss.item() * scan.size(0) # Ensure the criterion reduction parameter is 'mean'

                    acc_IoU += sum_IoU(pred_mask3d, mask3d)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_IoU = acc_IoU / dataset_sizes[phase]
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
                
            print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')
        print(f'Best validation mean IoU: {best_IoU:4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only = True))
    return model

if __name__ == '__main__':
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
        print(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    # Define transforms.
    # Augments are random changes, which are useful for training but not validation.
    transform = get_transform(data='input')
    target_transform = get_transform(data='target')
    augment = T.Compose([
        T.GaussianNoise(mean = 0, sigma = 0.2),
        T.RandomHorizontalFlip(p = 0.5),
        T.RandomVerticalFlip(p = 0.5),
        T.RandomRotation(degrees = 15),
        T.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
    ])

    # Set up datasets and dataloaders
    data_dir = 'data'
    image_datasets = {x : MRIDataset(os.path.join(data_dir, x), x, transform, target_transform, augment) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # Improved DataLoader configuration for better GPU utilization
    num_workers = min(4, os.cpu_count())  # Use multiple workers for async data loading
    batch_size = 1  # Keep batch_size = 1 due to variable scan sizes and small validation set
    dataloaders = {x: DataLoader(
        image_datasets[x], 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=torch.cuda.is_available(),  # Pin memory for faster GPU transfer
        collate_fn=custom_collate_fn  # Handle variable-sized 3D volumes
    ) for x in ['train', 'val']}
    
    # Initialize model, loss, optimizer, and scheduler
    model = get_model_instance_segmentation(num_classes = 2, device = device, trained = False)
    criterion = Combined_Loss(device, alpha = 0.5, beta = 0.7, gamma = 0.75, ce_weights=(0.1, 0.9))
    
    # Use AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr = 0.0001, weight_decay = 0.01)
    
    # Conservative learning rate schedule
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-7
    )

    print(f"\nDataset sizes: {dataset_sizes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    model = train(model, device, criterion, optimizer, dataloaders, lr_scheduler, dataset_sizes, 
                  num_epochs=100, patience=20)

    # Save the model parameters
    torch.save(model.state_dict(), 'model_params.pth')