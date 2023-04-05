import torch
from torch.utils.data import Dataset


class FBPDataset(Dataset):
    """ For Supervised Learning Pipeline """
    def __init__(self, tokenizer, df):
        super().__init__()
        self.tokenizer = tokenizer  # get from train.py
        self.df = df

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
    """ For Semi-Supervised Learning, Meta Pseudo Labels Pipeline """
    def __init__(self, tokenizer, df):
        super().__init__()
        self.df = df
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


