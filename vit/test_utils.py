import torch

"""
Test Loop for Evaluation

Runs inference on the test dataset and computes the total number of correct predictions.

Args:
    vit (nn.Module): Trained Vision Transformer model.
    test_dataloader (DataLoader): DataLoader for the test dataset.
    device (torch.device): Target device ('cuda' or 'cpu').

Returns:
    correct (int): Total number of correct predictions.
    total (int): Total number of samples evaluated.

Note:
    Uses torch.autocast for mixed-precision inference if CUDA is available.
    Final accuracy is correct / total.
"""

def test(vit, test_dataloader, device):
    correct, total = 0, 0

    with torch.no_grad(): 
        for image, label in test_dataloader:
            image, label = image.to(device), label.to(device)

            if torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda"):
                    output = vit(image)
            else:
                output = vit(image)

            pred = torch.argmax(output, dim=1)
            total += label.size(0)
            correct += (pred == label).sum().item()

    return correct, total




