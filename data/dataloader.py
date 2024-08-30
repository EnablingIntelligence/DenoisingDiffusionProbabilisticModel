from typing import Union, Tuple

from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose


def get_mnist_loader(
        train: bool,
        batch_size: int,
        normalize: bool = True,
        data_path: str = "./data",
        shuffle: bool = True,
        num_workers: int = 1
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    transforms = [
        ToTensor()
    ]

    if normalize:
        transforms.append(
            Normalize((0.1307,), (0.3081,))
        )

    transform = Compose(transforms)

    if train:
        dataset = MNIST(data_path, train=True, transform=transform, download=True)
        train_dataset, val_dataset = random_split(dataset, [50_000, 10_000])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_loader, val_loader

    test_dataset = MNIST(data_path, train=False, transform=transform, download=True)
    return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_cifar10_loader(
        train: bool,
        batch_size: int,
        normalize: bool = True,
        data_path: str = "./data",
        shuffle: bool = True,
        num_workers: int = 1
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    transforms = [
        ToTensor()
    ]

    if normalize:
        transforms.append(
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )

    transform = Compose(transforms)

    if train:
        dataset = CIFAR10(data_path, train=True, transform=transform, download=True)
        train_dataset, val_dataset = random_split(dataset, [42_000, 8_000])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_loader, val_loader

    test_dataset = CIFAR10(data_path, train=False, transform=transform, download=True)
    return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
