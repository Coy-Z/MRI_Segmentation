# Checking the loss function behaves as expected on the most basic example.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from utils.fcn_resnet101_util import sum_IoU, Combined_Loss

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Module()  # Dummy model for testing

criterion = Combined_Loss(device)

mask = torch.Tensor([ # 2 * 1 * 2
    [[1, 0]],
    [[0, 0]]
]).long().to(device = "cuda")
pred_logits = torch.Tensor([ # 2 * 2 * 1 * 2
    [[[3, 3]],
     [[15, 15]]],
    [[[15, 15]],
     [[3, 3]]]
]).to(device = "cuda")
pred_mask = torch.Tensor([ # 2 * 1 * 2
    [[1, 1]],
    [[0, 0]]
]).to(device = "cuda")

loss = criterion(pred_logits, mask)
print(loss.item())

print(sum_IoU(mask, pred_mask))