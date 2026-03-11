"""
This file contains functions to create a PyTorch dataset and PyTorch dataloader
for data.
"""
import torch
from torchvision.datasets import ImageFolder, Subset
from torchvision import transforms
from torch.utils.data import Subset, Dataset, random_split, DataLoader

def create_dataset(root: str, transform: transforms.Compose) -> ImageFolder:
    """Creates a dataset PyTorch dataset using images in given path.
    Creates an ImageFolder to create a dataset from images.

    Args:
        root (str): Path to the directory that contains food category directories.
        transforms (transforms.Compose): Transformations to apply the images.
    Returns:
        An ImageFolder object that represents the dataset.
    """
    dataset = ImageFolder(
            root=root,
            transform=transform)

    return dataset

def split_dataset_into_train_val_test(dataset: Dataset, train_rate: float, val_rate: float, test_rate: float, seed: int|None=None) -> tuple[Subset, Subset, Subset]:
    """Splits the given dataset into three subsets.
    Creates a train, validation, and test datasets from given single dataset using the specified
    rates for train, validation, and test datasets.

    Args:
        dataset (Dataset): Dataset to be splitted.
        train_rate (float): The rate of the train dataset (e.g 0.5 -> the half of the dataset will be train)
        val_rate (float): Rate of the validation dataset
        test_rate (float): Rate of the test dataset
        seed (int): An integer for reproducibility
    Returns:
        A tuple of three Subset object that represents train, validation, and test datasets in order.

        Example usage:
            train_dataset, val_dataset, test_dataset = split_dataset_into_train_val_test(
                                                           my_dataset, 0.7, 0.15, 0.15)

            -> 70% train, 15% val, 15% test dataset.
    """
    if seed:
        generator = torch.Generator().manual_seed(42)
    else:
        generator = None
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_rate, val_rate, test_rate], generator=generator)
    
    return train_dataset, val_dataset, test_dataset

def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a PyTorch DataLoader.
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader

def build_dataloaders(root: str,
               transform: transforms.Compose,
               rates: tuple[float, float, float]= (0.7, 0.15, 0.15),
               batch_size: int=32,
               shuffle: bool=True,
               seed: int=42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test dataloaders.
    Loads the image data into an ImageFolder from given path. Then splits the data
    into train, validation, and test sets. Then creates dataloaders for each.

    Args:
        root (str): Path to directory that contains food category directories.
        transform (transforms.Compose): Transformations to apply to images.
        rates (tuple[float, float, float]): Split rates for train, validation, and test data. (e.g (0.7, 0.15, 0.15) -> (train_percent, val_percent, test_percent))
        batch_size (int): An integer that represents the batch size.
        shuffle (bool): Shuffle the images in train data?
        seed (int): seed for reproducibility
    Returns:
        A tuple of three DataLoader's.
        The first one is train dataloader, second one is validation and the last one is test dataloader.
    """
    dataset = create_dataset(root=root, transform=transform)

    train_dataset, val_dataset, test_dataset = split_dataset_into_train_val_test(
                                            dataset, rates[0], rates[1], rates[2])

    train_dataloader = create_dataloader(train_dataset, batch_size, shuffle)
    val_dataloader = create_dataloader(val_dataset, batch_size, shuffle)
    test_dataloader = create_dataloader(test_dataset, batch_size, shuffle)

    return train_dataloader, val_dataloader, test_dataloader
