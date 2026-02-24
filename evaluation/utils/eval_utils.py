import os
from sys import path
import pandas as pd
import pydicom
import torch
import torchxrayvision as xrv
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def parse_pediatric_dataset(
    base_path: str,
    save_root: str,
    pathologies: list[str],
    is_train = True
):
    root_train = os.path.join(save_root, "train")
    root_test = os.path.join(save_root, "test")

    os.makedirs(root_train, exist_ok=True)
    os.makedirs(root_test, exist_ok=True)

    if is_train:
        csv_path_label = os.path.join(base_path, "image_labels_train.csv")
    else:
        csv_path_label = os.path.join(base_path, "image_labels_test.csv")

    labels_df = pd.read_csv(csv_path_label)
    labels_df = labels_df[["image_id"] + pathologies].set_index('image_id')

    train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42)
    train_ids = set(train_df.index)

    train_metadata = []
    test_metadata = []

    for file in os.listdir(base_path):
        if file.endswith('.dicom'):
            img_id = file.replace('.dicom', '')
            file_path = os.path.join(base_path, file)
            ds = pydicom.dcmread(file_path)

            img_np = ds.pixel_array

            header_max = (2**ds.BitsStored) - 1
            actual_max = img_np.max()
            maxval = max(header_max, actual_max)
            img_np = xrv.datasets.normalize(img_np, maxval)
                    
            img_np = img_np[None, :, :]
            img_np = xrv.datasets.XRayCenterCrop()(img_np)
            img_np = xrv.datasets.XRayResizer(224)(img_np)
            
            img_tensor = torch.from_numpy(img_np)

            entry = labels_df.loc[img_id].to_dict()
            entry['image_id'] = img_id

            if img_id in train_ids:
                torch.save(img_tensor, os.path.join(root_train, f"{img_id}.pt"))
                train_metadata.append(entry)
            else:
                torch.save(img_tensor, os.path.join(root_test, f"{img_id}.pt"))
                test_metadata.append(entry)

    # Save dedicated manifest files
    pd.DataFrame(train_metadata).to_csv(os.path.join(root_train, "labels.csv"), index=False)
    pd.DataFrame(test_metadata).to_csv(os.path.join(root_test, "labels.csv"), index=False)


class PediatricXRDataset(Dataset):
    def __init__(self, split, base_path, transform=None):
        
        if split == 'train':
            self.data_root = base_path + r"\train"
            label_csv = base_path + r"\train\labels.csv"

        else:
            self.data_root = base_path + r"\test"
            label_csv = base_path + r"\test\labels.csv"

        self.labels_df = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        """Return total number of pediatric samples."""
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx]['image_id']
        label_vector = self.labels_df.iloc[idx, 0:-1].values.astype('float32')

        img_path = os.path.join(self.data_root, f"{img_id}.pt")
        img_tensor = torch.load(img_path)

        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return img_tensor, torch.tensor(label_vector)