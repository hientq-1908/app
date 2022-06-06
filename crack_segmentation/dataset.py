from torch.utils.data import Dataset
import torch
import albumentations as A
import glob
from PIL import Image
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class CrackDataset(Dataset):
    def __init__(self, img_dir, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode
        self.label_paths = None
        if mode == 'train':
            self.img_paths = glob.glob('{}/{}/image/*.png'\
                .format(img_dir, mode))
            self.label_paths = glob.glob('{}/{}/label/*.png'\
                .format(img_dir, mode))
            assert len(self.img_paths) == len(self.label_paths)
        else:
            self.img_paths = glob.glob('{}/{}/*.png'\
                .format(img_dir, mode))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            image = cv2.imread(self.img_paths[index])
            mask = cv2.imread(self.label_paths[index])
            # this images have 4 channels, only take 3
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            transform = self._get_transform().get(self.mode)
            transformed = transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            # this masks have channels, and various manigtude
            # convert to binary mask
            transformed_mask = np.where(transformed_mask>0, 1, 0)[:, :, 0]
            return transformed_image, transformed_mask
        else:
            image = cv2.imread(self.img_paths[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            transform = self._get_transform().get(self.mode)
            transformed = transform(image=image)
            return transformed['image']

    def _get_transform(self):
        transform = {
            'train': A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(width=224, height=224),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            'test': A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        }
        return transform

    def reverse_tranform(self, image):
        image = image.cpu().numpy().transpose(1,2,0)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = std * image + mean
        image = np.clip(image, 0, 1)
        image *= 225
        return image.astype(np.uint8)
