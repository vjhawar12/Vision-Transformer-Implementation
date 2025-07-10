from torchvision import datasets

def get_training_data(train_transform):
    training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    return training_data

def get_test_data(test_transform):
    test_data = datasets.CIFAR10(root="data", train=False, download=False,  transform=test_transform)
    return test_data

