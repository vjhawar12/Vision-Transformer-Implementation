import torch
import tqdm
from torchvision.transforms.v2 import RandomChoice

"""
  Training loop for a single epoch using mixed-precision training and data augmentation.

    Args:
        vit (nn.Module): Vision Transformer model.
        device (torch.device): Training device (usually CUDA).
        cutmix_or_mixup_start (int): Epoch after which to randomly apply either CutMix or MixUp.
        mixup_start (int): Epoch after which to optionally apply only MixUp.
        max_cutmix_or_mixup (float): Maximum probability for CutMix or MixUp.
        max_mixup (float): Maximum probability for MixUp.
        optim (Optimizer): Optimizer instance.
        mixup (callable): MixUp augmentation function.
        cutmix (callable): CutMix augmentation function.
        sft_loss_fn (callable): Soft-label loss function (e.g., KLDiv with soft targets).
        loss_fn (callable): Standard classification loss function (e.g., CrossEntropyLoss).
        scaler (GradScaler): PyTorch AMP scaler for mixed-precision training.
        i (int): Current epoch number.
        loop (tqdm.tqdm): tqdm progress bar wrapping the training dataloader.

    Returns:
        total (int): Total number of examples processed.
        correct (int): Number of correct predictions.
        avg_loss (float): Average training loss for the epoch.
"""


def train(vit, device, cutmix_or_mixup_start, mixup_start, max_cutmix_or_mixup, max_mixup, optim, mixup, cutmix, sft_loss_fn, loss_fn, scaler, i, loop):
    running_loss, total, correct = 0, 0, 0

    for j, (input, labels) in enumerate(loop, 1):
        input, labels = input.to(device), labels.to(device)
        optim.zero_grad()

        # beginning to introduce CutMix at the cutmix_or_mixup_start epoch. There is a bias towards MixUp to limit CutMix since it is a severe augmentation.
        if i >= cutmix_or_mixup_start and torch.rand(1) < min(max_cutmix_or_mixup, i/100):
            cutmix_or_mixup = RandomChoice([cutmix, mixup], p=[0.3, 0.7])
            input, labels = cutmix_or_mixup(input, labels)

        # mixup_start is intended to be < cutmix_or_mixup_start because the model should be gently introduced to trickier data in order to not confuse it too much
        elif i >= mixup_start and torch.rand(1) < min(max_mixup, i/100):
            input, labels = mixup(input, labels)

        # autocasting to lower precision types when possible for speed
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = vit(input)

            # SoftCrossEntropy returns per-sample loss for soft labels. We take the mean over the batch.
            if labels.ndim == 2:
                # sft_loss_fn is SoftCrossEntropy
                loss = sft_loss_fn(output, labels)
                running_loss += loss.mean().item()
            else:
                # loss_fn is regular CrossEntropyLoss
                loss = loss_fn(output, labels)
                running_loss += loss.item()

            # scaling gradients often boosts convergence
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        avg_loss = running_loss / j
        loop.set_postfix(loss=avg_loss)

        pred = torch.argmax(output, dim=1)
        
        if labels.ndim == 2:
            labels = labels.argmax(dim=1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

    return total, correct, avg_loss

"""
  Validation loop for a single epoch using mixed-precision training.

    Args:
        vit (nn.Module): Vision Transformer model.
        device (torch.device): Training device (usually CUDA).
        test_dataloader (torch.utils.data.DataLoader): DataLoader for test data

    Returns:
        val_total (int): Total number of examples processed.
        val_correct (int): Number of correct predictions.
"""

def validate(vit, test_dataloader, device):
    vit.eval()
    val_total, val_correct = 0, 0

    with torch.no_grad():
        for image, label in test_dataloader:
            image, label = image.to(device), label.to(device)

            # autocasting to lower precision types when possible for speed
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
              output = vit(image)

            pred = torch.argmax(output, dim=1)
            val_total += label.size(0)
            val_correct += (pred == label).sum().item()

        return val_total, val_correct

