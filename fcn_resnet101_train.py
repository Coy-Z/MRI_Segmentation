import time
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from tempfile import TemporaryDirectory
from utils.fcn_resnet101_util import clip_and_scale, get_model_instance_segmentation, sum_IoU, get_transform, MRIDataset, Combined_Loss

def train(model, device, criterion, optimizer, dataloaders, scheduler, num_epochs):
    """
    Trains the model and returns the best model based on validation IoU.

    Args:
        model: The segmentation model to train.
        device: Device to use ('cuda' or 'cpu').
        criterion: Loss function.
        optimizer: Optimizer.
        dataloaders: Dict of DataLoader objects for 'train' and 'val'.
        scheduler: Learning rate scheduler.
        num_epochs: Number of epochs to train.

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

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

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

                    # Move data to GPU - Also dataloader batches, adding another dimension
                    scan = scan.to(device).squeeze(0)
                    mask3d = mask3d.to(device).squeeze(0).long()

                    # Forward pass: Track history if in training phase
                    # Note: model expects a batch dimension, so scan is (1 * D * H
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(scan)
                        pred_mask3d_logits = outputs['out']
                        pred_mask3d = torch.argmax(outputs['out'], dim = 1)
                        loss = criterion(pred_mask3d_logits, mask3d)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Accumulate Statistics
                    running_loss += loss.item() * scan.size(0) # Ensure the criterion reduction parameter is 'mean'

                    acc_IoU += sum_IoU(pred_mask3d, mask3d)
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_IoU = acc_IoU / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Mean IoU: {epoch_IoU:.4f}')

                # Deep copy the model
                if phase == 'val' and epoch_IoU > best_IoU:
                    best_IoU = epoch_IoU
                    torch.save(model.state_dict(), best_model_params_path)
            print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')
        print(f'Best validation mean IoU: {best_IoU:4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only = True))
    return model

if __name__ == '__main__':
    # Select device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f'Using {device} device.')

    # Define transforms.
    # Augments are random flips, which are useful for training but not validation.
    transform = get_transform(data='input')
    target_transform = get_transform(data='target')
    augment = T.Compose([
        T.RandomHorizontalFlip(p = 0.5),
        T.RandomVerticalFlip(p = 0.5)
    ])

    # Set up datasets and dataloaders
    data_dir = 'data'
    image_datasets = {x : MRIDataset(os.path.join(data_dir, x), x, transform, target_transform, augment) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size = 1, shuffle = True, num_workers = 0, persistent_workers = False) for x in ['train', 'val']}
    
    # Initialize model, loss, optimizer, and scheduler
    model = get_model_instance_segmentation(num_classes = 2)
    criterion = Combined_Loss(device, alpha = 2, beta = 0.8, gamma = 0.75, ce_weights=(0.1, 1))
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 60, gamma = 0.1)

    # Train the model
    model = train(model, device, criterion, optimizer, dataloaders, lr_scheduler, num_epochs = 100)

    # Save the model parameters
    torch.save(model.state_dict(), 'model_params.pth')
