from torch.utils.data import DataLoader


def get_dataloaders(training_data, test_data, batchsize):
    train_dataloader = DataLoader(dataset=training_data, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=8)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=False)

    return train_dataloader, test_dataloader


