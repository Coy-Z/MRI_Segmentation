import torch
from fcn_resnet101_util import sum_IoU, CE_DICE_Loss

criterion = CE_DICE_Loss(1)

mask = torch.Tensor([ # 2 * 1 * 2
    [[1, 0]],
    [[0, 0]]
]).long()
pred_logits = torch.Tensor([ # 2 * 2 * 1 * 2
    [[[3, 3]],
     [[15, 15]]],
    [[[15, 15]],
     [[3, 3]]]
])
pred_mask = torch.Tensor([ # 2 * 1 * 2
    [[1, 1]],
    [[0, 0]]
])

loss = criterion(pred_logits, mask)
print(loss.item())

print(sum_IoU(mask, pred_mask))