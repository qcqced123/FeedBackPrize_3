import torch
from torch.utils.data import Dataset

from text_preprocessing import *
from base import BaseDataLoader


class FBPDataset(Dataset):
    """
    For Supervised Learning Pipeline
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.df = text_preprocess(load_data('../data/FB3_Dataset/train.csv'))
        self.tokenizer = tokenizer

    def tokenizing(self, text):
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors=None,
            add_special_tokens=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v)
        return inputs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = self.tokenizing(self.df.iloc[idx, 1])
        label = torch.tensor(self.df.iloc[idx, 2:8], dtype=torch.float)
        return inputs, label


class MPLDataset(Dataset):
    """
    For Semi-Supervised Learning, Meta Pseudo Labels Pipeline
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.df = pseudo_dataframe(
            fb1_preprocess(load_data('../data/FB1_Dataset/train.csv')),
            fb2_preprocess(load_data('../data/FB2_Dataset/train.csv')),
        )
        self.tokenizer = tokenizer

    def tokenizing(self, text):
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors=None,
            add_special_tokens=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v)
        return inputs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = self.tokenizing(self.df.iloc[idx, 1])
        return inputs


