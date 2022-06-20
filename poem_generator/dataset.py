from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

class PoemDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1000):
        super().__init__()
        self.input_ids = list()
        self.attn_masks = list()
        for i in data:
            encoded_text = tokenizer(
                '<BOS>' + i + '<EOS>',
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.input_ids.append(encoded_text.input_ids)
            self.attn_masks.append(encoded_text.attention_mask)
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
    
