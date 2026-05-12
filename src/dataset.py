"""
Dataset and DataLoader implementation for AI for Computational Mechanics (Assignment 1).
Maps individual .npy image files to their E_eff labels using the provided CSV file.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MicrostructureDataset(Dataset):
    def __init__(self, dataframe, img_dir):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'filename' and 'E_eff'.
            img_dir (str): Path to the directory containing the .npy image files.
        """
        # Reset index to ensure __getitem__ works seamlessly after train_test_split
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get filename and target label from the current row
        row = self.df.iloc[idx]
        filename = row['filename']
        label = row['E_eff']
        
        # Ensure the filename ends with .npy (just in case the CSV omits the extension)
        if not filename.endswith('.npy'):
            filename += '.npy'
            
        # 2. Load the individual .npy image
        img_path = os.path.join(self.img_dir, filename)
        image = np.load(img_path)
        
        # 3. Convert image to tensor and add the channel dimension (1, 65, 65)
        # CNNs require (Channels, Height, Width)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        # 4. Convert label to tensor shape (1,)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return image_tensor, label_tensor


def create_dataloaders(img_dir, labels_path, batch_size=64, train_ratio=0.8, val_ratio=0.1, max_samples=None):
    """
    Loads the CSV, optionally slices it for efficiency ranking, splits it, 
    and returns DataLoaders pointing to the image directory.
    """
    # 1. Load the labels CSV
    df = pd.read_csv(labels_path)
    
    # Optional: Verify columns exist to prevent silent bugs
    assert 'filename' in df.columns and 'E_eff' in df.columns, "CSV must contain 'filename' and 'E_eff' columns!"

    # 2. Limit dataset size to improve the "Efficiency Score" ranking
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42)

    # 3. Calculate split sizes
    test_ratio = 1.0 - train_ratio - val_ratio

    # 4. Perform the splits on the DataFrame
    df_temp, df_test = train_test_split(df, test_size=test_ratio, random_state=42)
    
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(df_temp, test_size=relative_val_ratio, random_state=42)

    # 5. Initialize PyTorch Datasets
    train_dataset = MicrostructureDataset(df_train, img_dir)
    val_dataset = MicrostructureDataset(df_val, img_dir)
    test_dataset = MicrostructureDataset(df_test, img_dir)

    # 6. Initialize PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data Pipeline Ready:")
    print(f"Images Directory: {img_dir}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
