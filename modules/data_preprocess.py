"""
creates dataoaders from dir
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
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
        transform=transform,
        target_transform=None
    )
    
    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=transform,
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
