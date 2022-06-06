from pickletools import optimize
from regex import P
from zmq import device
from dataset import CrackDataset
from network import ResUnet
import torch
import torch.nn as nn
from diceloss import DiceLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision.utils import make_grid

class Trainer:
    def __init__(self, model, train_set=None, eval_set=None, device='cpu'):
        self.learning_rate = 2e-4
        self.num_epochs = 40
        self.model = model
        self.train_set = train_set
        self.eval_set = eval_set
        self.batch_size = 4
        self.criterion = DiceLoss(epsilon=2e-4)
        self.device = device
        self.checkpoint = 'checkpoint/'
        self.cur_epoch = -1
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
    def _train_one_epoch(self, epoch, model, train_loader, eval_loader, optimizer, criterion, device, writer):
        model.train()
        train_loss = []
        eval_loss = []
        loop = enumerate(tqdm(train_loader, total=len(train_loader))) 
        for i, (images, masks) in loop:
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1) # add 1 class dim
            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            writer.add_scalar('training_loss', loss, epoch*len(train_loader)+i)
        ###################
        model.eval()
        loop = enumerate(tqdm(eval_loader, total=len(eval_loader)))
        with torch.no_grad():
            for i, (images, masks) in loop:
                images, masks = images.to(device), masks.to(device)
                masks = masks.unsqueeze(1) # 1 N class dim
                preds = model(images)
                loss = criterion(preds, masks)
                eval_loss.append(loss.item())
                writer.add_scalar('eval_loss', loss, epoch*len(eval_loader)+i)
        return np.mean(train_loss), np.mean(eval_loss)
    
    def train(self):
        assert self.train_set is not None
        assert self.eval_set is not None
        writer = SummaryWriter('runs/train')
        model = self.model.to(self.device)
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        eval_loader = DataLoader(
            self.eval_set,
            batch_size=self.batch_size,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            self.cur_epoch = epoch
            train_loss, eval_loss = self._train_one_epoch(
                epoch,
                model,
                train_loader,
                eval_loader,
                optimizer,
                self.criterion,
                self.device,
                writer
            )
            print('epoch {}/{}, train_loss {:.4f} eval_loss {:.4f}'.format(epoch+1, self.num_epochs, train_loss, eval_loss))

            if (epoch+1) % 5 == 0:
                self._save_checkpoint(self.checkpoint + f'epoch_{epoch+1}')
        writer.close()

    def _save_checkpoint(self, path):
        state_dict = {
            'model': self.model.state_dict(),
            'epoch': self.cur_epoch
        }
        torch.save(state_dict, path)
        print('saved')

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.cur_epoch = checkpoint['epoch']
        print('loaded')
    
    def predict(self, img_dir):
        writer = SummaryWriter('runs/predict')
        dataset = CrackDataset(img_dir, mode='test')
        test_loader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)
        model = self.model.to(self.device)
        for i, images in enumerate(test_loader):
            images = images.to(self.device)
            preds = model(images)
            probs = torch.sigmoid(preds)
            probs = torch.where(probs>0.5, 0, 1)
            masks = probs.expand_as(images)
            _images = torch.cat([images, masks.detach()], dim=0)
            _images = _images.cpu()
            grid = make_grid(_images, nrow=4, normalize=True)
            writer.add_image('predicted_images', grid, i)
        writer.close()