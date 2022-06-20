import torch
from torch.utils.data import random_split

def train_val_split(split, dataset):
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return train_set, val_set