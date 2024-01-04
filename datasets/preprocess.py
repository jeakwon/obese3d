import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader

def array2d_to_tensor3d(array2d):
    n_rows, n_cols = array2d.shape
    array3d = array2d.reshape(n_rows, n_cols//3, 3)
    array3d -= array3d.mean(axis=(0, 1)) # offset
    tensor3d = torch.from_numpy(array3d).float()
    return tensor3d

def preprocess(csv_path, seq_len=30, seed=42):
    # data
    df = pd.read_csv(csv_path, header=None)
    array2d = df.values
    tensor3d = array2d_to_tensor3d(array2d)
    data = tensor3d.split(seq_len, dim=0)
    if data[-1].shape[0] < seq_len:
        data = data[:-1]
    train, valid_test = train_test_split(data, test_size=0.2, random_state=seed)
    valid, test = train_test_split(valid_test, test_size=0.5, random_state=seed)

    # info
    basename = os.path.basename(csv_path)
    name, ext = os.path.splitext(basename)
    date, week, food, ID, weight = name.split('_')
    
    data = dict(
        train=train,
        valid=valid,
        test=test,
    )

    label = dict(
        name=name,
        date=date, 
        week=int(week), 
        food=food, 
        ID=ID, 
        weight=float(weight),
    )

    return data, label

class Obese3dDataset(Dataset):
    def __init__(self, data_dir, input_type='train', target_type='ID', seq_len=30, seed=42):
        """
        `Obese3dDataset` is a subclass of `torch.utils.data.Dataset` that is specifically designed to handle data for Obesity 3d action skeletons . 
        It processes CSV files from given data_dir, splits them into sequences, and prepares them for model training and evaluation.

        #### Initialization Parameters:

        - `data_dir` (str): The directory path containing the CSV files. Each file represents a set of data points.
        - `input_type` (str, default='train'): Specifies the type of data to be loaded. Options are 'train', 'valid', or 'test', corresponding to training, validation, and testing datasets, respectively.
        - `target_type` (str, default='ID'): Specifies the target label for the dataset. For example, 'ID' could be used for tasks involving identity recognition. Options are 'name', 'date', 'week', 'food', 'ID', 'weight'
        - `seq_len` (int, default=30): The length of the sequence to be generated from the data. This defines the size of each split from the original data.
        - `seed` (int, default=42): The seed value for random operations, ensuring reproducibility in splitting the dataset.
        """
        self.data_dir=data_dir
        self.input_type=input_type
        self.target_type=target_type
        self.seq_len=seq_len
        self.seed=seed
        self.label_encoder = LabelEncoder()

        datas = []
        labels = []
        for f in os.listdir(data_dir):
            csv_path = os.path.join(data_dir, f)
            data, label = preprocess(csv_path, seq_len=seq_len, seed=seed)
            datas += data[input_type]
            labels += [label[target_type]] * len(data[input_type])

        
        self.inputs = datas
        self.targets = self.label_encoder.fit_transform(labels)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __repr__(self):
        return f'[{self.__class__.__name__}]{self.input_type}(target={self.target_type}, seq_len={self.seq_len}, seed={self.seed})'

    def decode(self, targets):
        return self.label_encoder.inverse_transform(targets)

def get_obese3d_loaders(data_dir, batch_size=16, target_type='ID', seq_len=30, num_workers=2, seed=42):
    """
    `get_obesity3d_loaders` is a function that is specifically designed to handle data for Obesity 3d action skeletons. 
    It processes CSV files from given data_dir, splits them into sequences, and prepares them for model training and evaluation.

    #### Initialization Parameters:

    - `data_dir` (str): The directory path containing the CSV files. Each file represents a set of data points.
    - `batch_size` (int, default=16): Specifies the batch size.
    - `target_type` (str, default='ID'): Specifies the target label for the dataset. For example, 'ID' could be used for tasks involving identity recognition. Options are 'name', 'date', 'week', 'food', 'ID', 'weight'
    - `seq_len` (int, default=30): The length of the sequence to be generated from the data. This defines the size of each split from the original data.
    - `num_workers` (int, default=2): Number of workers for subprocess
    - `seed` (int, default=42): The seed value for random operations, ensuring reproducibility in splitting the dataset.

    #### Returns:

    - `train_loader` : 80% of sequence data used for model training
    - `valid_loader` : 10% of sequence data used for model validation
    - `test_loader` : 10% of sequence data used for model testing
    """
    train_set = Obese3dDataset(data_dir=data_dir, input_type='train', target_type=target_type, seq_len=seq_len, seed=seed)
    valid_set = Obese3dDataset(data_dir=data_dir, input_type='valid', target_type=target_type, seq_len=seq_len, seed=seed)
    test_set = Obese3dDataset(data_dir=data_dir, input_type='test', target_type=target_type, seq_len=seq_len, seed=seed)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader

if __name__ == "__main__": 
    # example useage
    train_loader, valid_loader, test_loader = get_obese3d_loaders(
        data_dir='/content/drive/MyDrive/collaboration/khw/data/coord', 
        batch_size=16, target_type='ID', seq_len=30, num_workers=2, seed=42)