from torch.utils.data import DataLoader

"""
Creates and returns PyTorch DataLoaders for training and testing datasets.

Args:
    training_data (Dataset): A PyTorch-compatible training dataset.
    test_data (Dataset): A PyTorch-compatible test dataset.
    batchsize (int, optional): Number of samples per batch. Defaults to 128.

Returns:
    Tuple[DataLoader, DataLoader]: 
        - train_dataloader with shuffling and pin_memory enabled.
        - test_dataloader without shuffling (for evaluation consistency).

Note:
    The training dataloader uses `num_workers=8` for parallel data loading, which can
    improve performance depending on system resources. 
"""

def get_dataloaders(training_data, test_data, batchsize=128):
    train_dataloader = DataLoader(dataset=training_data, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=8)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=False)

    return train_dataloader, test_dataloader


