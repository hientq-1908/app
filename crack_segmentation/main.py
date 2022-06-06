from dataset import CrackDataset
from trainer import Trainer
from network import ResUnet
import torch
from utils import dataset_split
from torch.utils.data import random_split

if __name__ == '__main__':
    img_dir = 'dataset\cracktile'
    train_ratio = 0.8
    dataset = CrackDataset(img_dir)
    train_len = int(len(dataset)*train_ratio)
    eval_len = len(dataset) - train_len
    train_set, eval_set = random_split(dataset, [train_len, eval_len])

    model = ResUnet(n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(
        model,
        train_set,
        eval_set,
        device
    )
    trainer.train()
    # trainer.load_checkpoint('checkpoint\epoch_40')
    # trainer.predict('dataset\crackconcrete')