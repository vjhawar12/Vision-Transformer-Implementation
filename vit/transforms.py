import torchvision

"""
Returns torchvision transform pipelines for training and testing data.

The training pipeline includes resizing, data augmentation, and normalization.
The test pipeline includes resizing and normalization only.
"""

def get_transforms():
    # Training pre-processing pipeline incorporates image resizing, random horizontal flip, random crop, color jitter, and normalization
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64), # Upscaling CIFAR-10 image from 32 to 64 increases the number of patches.
        torchvision.transforms.RandomHorizontalFlip(), 
        torchvision.transforms.RandomCrop(64, padding=4),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)) 
        # normalizing mean helps optims like Adam, scaling ensures uniformity across the inputs
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    return train_transform, test_transform
