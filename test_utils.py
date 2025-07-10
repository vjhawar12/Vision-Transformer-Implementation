import torch

def test(vit, test_dataloader, device):
    correct, total = 0, 0

    with torch.no_grad():
        for image, label in test_dataloader:
            image, label = image.to(device), label.to(device)

            with torch.amp.autocast():
                output = vit(image)

            pred = torch.argmax(output, dim=1)
            total += label.size(0)
            correct += (pred == label).sum().item()

    return correct, total




