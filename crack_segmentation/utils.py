import datasets
from torch.utils.data import random_split
from dataset import CrackDataset

def dataset_split(img_dir, train_ratio=0.8):
    dataset = CrackDataset(img_dir)
    train_len = int(len(dataset)*train_ratio)
    eval_len = len(dataset) - train_len
    train_set, eval_set = random_split(dataset, [train_len, eval_len])
    return train_set, eval_set

