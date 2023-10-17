"""
creates dataoaders from dir
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import torch
from torch import nn

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transforms: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """creates train and test dataloaders

    Args:
        train_dir (str): _description_
        test_dir (str): _description_
        transform (transforms.Compose): _description_
        batch_size (int): _description_
        num_workers (int, optional): _description_. Defaults to NUM_WORKERS.

    Returns:
        Tuple(train_dataloader, test_dataloader, class_names): _description_
    """
    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=transforms,
        target_transform=None
    )
    
    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=transforms,
        target_transform=None
    )
    
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True
        )
    
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        shuffle=False)
    
    class_names = train_data.classes
    
    return train_dataloader, test_dataloader, class_names


def split_dataset(dataset:torchvision.datasets, split_size:float=0.2, seed:int=42):
    """
    randomly splits a given dataset into two proportions based on split_size and seed.

    Args:
        dataset (torchvision.datasets): A PyTorch Dataset, typically one from torchvision.datasets.
        split_size (float, optional): How much of the dataset should be split? 
            E.g. split_size=0.2 means there will be a 20% split and an 80% split. Defaults to 0.2.
        seed (int, optional): Seed for random generator. Defaults to 42.

    Returns:
        tuple: (random_split_1, random_split_2) where random_split_1 is of size split_size*len(dataset) and 
            random_split_2 is of size (1-split_size)*len(dataset).
    """
    # Create split lengths based on original dataset length
    length_1 = int(len(dataset) * split_size) # desired length
    length_2 = len(dataset) - length_1 # remaining length
        
    # Print out info
    print(f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {length_1} ({int(split_size*100)}%), {length_2} ({int((1-split_size)*100)}%)")
    
    # Create splits with given random seed
    random_split_1, random_split_2 = torch.utils.data.random_split(dataset, 
                                                                   lengths=[length_1, length_2],
                                                                   generator=torch.manual_seed(seed)) # set the random seed for reproducible splits
    return random_split_1, random_split_2
