import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tempfile import TemporaryDirectory
from fcn_resnet101_util import clip_and_scale, get_model_instance_segmentation, sum_IoU, MRIDataset, CE_Dice_Loss

def train(model, device, criterion, optimizer, dataloaders, scheduler, num_epochs):
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

                    # Forward
                    # Track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(scan)
                        pred_mask3d_logits = outputs['out']
                        pred_mask3d = torch.argmax(outputs['out'], dim = 1)
                        loss = criterion(pred_mask3d_logits, mask3d)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * scan.size(0) # Ensure the criterion reduction parameter is 'mean'

                    acc_IoU += sum_IoU(pred_mask3d, mask3d) # Complete
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
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f'Using {device} device.')

    # Define transforms. Training transform is transform and augment. Validation transform is just transform.
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale = True),
        T.Resize(size = (41, 41), interpolation=T.InterpolationMode.BILINEAR),
        T.Lambda(clip_and_scale)
    ])
    target_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale = True),
        T.Resize(size = (41, 41), interpolation=T.InterpolationMode.NEAREST),
        T.Lambda(clip_and_scale)
    ])
    augment = T.Compose([
        T.RandomHorizontalFlip(p = 0.5),
        T.RandomVerticalFlip(p = 0.5)
    ])

    # Set up datasets
    data_dir = 'data'
    image_datasets = {x : MRIDataset(os.path.join(data_dir, x), x, transform, target_transform, augment) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size = 1, shuffle = True, num_workers = 0, persistent_workers = False) for x in ['train', 'val']}

    model = get_model_instance_segmentation(num_classes = 2)

    criterion = CE_Dice_Loss(device, alpha = 0.1, beta = 0.7, gamma = 0.75)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.2)

    model = train(model, device, criterion, optimizer, dataloaders, lr_scheduler, num_epochs = 100)

    torch.save(model.state_dict(), 'model_params.pth')
