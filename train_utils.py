import torch
import tqdm
from torchvision.transforms.v2 import RandomChoice


def train(vit, device, cutmix_or_mixup_start, mixup_start, max_cutmix_or_mixup, max_mixup, optim, mixup, cutmix, sft_loss_fn, loss_fn, scaler, i, loop):
    total, correct = 0, 0
    
    for j, (input, labels) in enumerate(loop, 1):
        input, labels = input.to(device), labels.to(device)
        optim.zero_grad()

        if i >= cutmix_or_mixup_start and torch.rand(1) < min(max_cutmix_or_mixup, i/100):
            cutmix_or_mixup = RandomChoice([cutmix, mixup], p=[0.3, 0.7])
            input, labels = cutmix_or_mixup(input, labels)

        elif i >= mixup_start and torch.rand(1) < min(max_mixup, i/100):
            input, labels = mixup(input, labels)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = vit(input)

            if labels.ndim == 2:
                loss = sft_loss_fn(output, labels)
                running_loss += loss.mean().item()
            else:
                loss = loss_fn(output, labels)
                running_loss += loss.item()

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

        return correct, total


def validate(vit, test_dataloader, device):
    vit.eval()
    val_total, val_correct = 0, 0

    with torch.no_grad():
        for image, label in test_dataloader:
            image, label = image.to(device), label.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
              output = vit(image)

            pred = torch.argmax(output, dim=1)
            val_total += label.size(0)
            val_correct += (pred == label).sum().item()

        return val_correct, val_total

