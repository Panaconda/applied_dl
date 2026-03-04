from abc import ABC, abstractmethod
import argparse
import json
from logging import root
from math import isnan
from multiprocessing import Pool
import os
from pathlib import Path
import re
import shutil
from typing import List, Final, Optional, Dict, Any
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageMath
from PIL import ImageFile
from pydicom import dcmread
from torchvision.transforms import Resize, Compose, CenterCrop
from tqdm import tqdm

import time
from prepare_pcxr.config import cfg

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRANSFORMS: Final = Compose([Resize(1024), CenterCrop(1024)])


# UTILS
# --------------------------------------------------------------------------------------
def read_file(file_path: str) -> List[str]:
    """Read a generic file line by line."""
    with open(file_path, 'r') as f:
        return [line.replace('\n', '') for line in f.readlines()]


def save_as_json(dictionary: Any, target: str) -> None:
    """Save a python object as JSON file."""
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)


class BaseParser(ABC):
    """Base class for parsing chest x-ray datasets."""

    def __init__(
        self,
        root: str,
        target_root: str,
        is_train: bool = True,
        transforms: Optional[Compose] = None,
        num_workers: int = 16,
        frontal_only: bool = True,
        *args,
        **kwargs
    ) -> None:
        """Initialize base parser."""
        self.root = root
        self.target_root = target_root
        self.is_train = is_train
        self.transforms = transforms
        self.num_workers = num_workers
        self.frontal_only = frontal_only

    @property
    @abstractmethod
    def keys(self) -> List[str]:
        """Identifier for image files."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the dataset."""
        pass

    @abstractmethod
    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        pass

    def __len__(self):
        """Return length of the dataset."""
        return len(self.keys)

    @abstractmethod
    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        pass

    def _get_image(self, key: str) -> Image:
        """Load and process an image for a given key."""
        img = Image.open(self._get_path(key))
        if img.mode == 'I':
            img = ImageMath.eval('img >> 8', img=img)
        img = img.convert('RGB')
        if self.transforms is not None:
            img = TRANSFORMS(img)
        return img

    def _process_idx(self, idx: int) -> Dict:
        """Worker function for parsing a dataset element."""
        key = self.keys[idx]

        # Define new file name and folder structure over 6-digit identifier.
        # Images are grouped in directories with 10k images each.
        # For example the image with corresponding to index 54321
        # will be placed in "{self.target_root}/05/'054321.jpg".
        file_id = str(idx).zfill(6)
        file_path = os.path.join(self.target_root, file_id + '.jpg')

        img = self._get_image(key)
        if img is None:
            return {}
        img.save(file_path, quality=95)

        meta_dict = {'path': os.path.abspath(file_path), 'key': key}

        meta_dict.update(self._get_meta_data(key))
        return {file_id: meta_dict}

    def parse(self, chunk_size: int = 64) -> None:
        """Parse the dataset."""
        index_dict = {}

        # Create all necessary directories.
        os.makedirs(self.target_root, exist_ok=True)

        # # Iterate over every entry in multiprocessing fashion.
        with Pool(processes=self.num_workers) as p:
            with tqdm(total=len(self), leave=False) as pbar:
                for entry in p.imap(
                    self._process_idx, range(0, len(self)), chunksize=chunk_size
                ):
                    index_dict.update(entry)
                    pbar.update()

        save_as_json(index_dict, target=os.path.join(self.target_root, 'index.json'))


# pcxr_png - Pediatric Chest X-Ray
# --------------------------------------------------------------------------------------
class PCXRParser(BaseParser):
    """Parser object for pcxr_png."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize pcxr_png parser."""
        super().__init__(*args, **kwargs)
        
        self.img_dir = os.path.join(self.root, 'train' if self.is_train else 'test')
        files = [f for f in os.listdir(self.img_dir) if f.endswith('.dicom')]

        #load csv
        annots_df = pd.read_csv(os.path.join(self.img_dir, 'annotations_train.csv' if self.is_train else 'annotations_test.csv'))
        labels_df = pd.read_csv(os.path.join(self.img_dir, 'image_labels_train.csv' if self.is_train else 'image_labels_test.csv'))

        data_list = []

        for file in tqdm(files, desc='Loading metadata'):

            img_id = file.split('.')[0]
            
            findings = annots_df[annots_df['image_id'] == img_id]['class_name'].unique()
            annotation = ", ".join(findings) if len(findings) > 0 else None
            
            subject_labels = labels_df[labels_df['image_id'] == img_id]

            if not subject_labels.empty:
                label_vec = subject_labels.iloc[0, 2:].values.astype(int) 
                active_pathologies = subject_labels.columns[2:][subject_labels.iloc[0, 2:] == 1].tolist()
                label_str = ", ".join(active_pathologies) if active_pathologies else "no finding"
            else:
                print(f"Warning: No label found for image_id {img_id}. Skipping.")
                continue

            data_list.append({
                    'image_id': file,
                    'annotation': annotation,
                    'label_vec': label_vec,
                    'label_pathologies': label_str
                })

        df = pd.DataFrame(data_list)

        self._keys = df['image_id'].tolist()

        prefix = "Findings: Frontal radiograph of a child."

        has_findings = (df['annotation'].notna()) & (df['annotation'] != "No finding")

        findings_abnormal = " Evaluation reveals " + df['annotation'] + "."
        findings_normal = " No acute radiographic abnormalities are observed."

        middle_text = np.where(has_findings, findings_abnormal, findings_normal)

        suffix = " Impressions: " + df['label_pathologies'] + "."

        df['report_text'] = prefix + middle_text + suffix

        self.meta_dict = {
            row['image_id']: {
                'report': row['report_text'],
                'label_vec': row['label_vec'].tolist() 
            } 
            for _, row in df.iterrows() 
        }

    def parse(self, chunk_size: int = 64) -> None:
        """Parse the dataset and copy labels."""
        os.makedirs(self.target_root, exist_ok=True)
        
        # Copy CSV files
        csv_files = [
            'annotations_train.csv' if self.is_train else 'annotations_test.csv',
            'image_labels_train.csv' if self.is_train else 'image_labels_test.csv'
        ]
        
        for csv_file in csv_files:
            src = os.path.join(self.img_dir, csv_file)
            dst = os.path.join(self.target_root, csv_file)
            if os.path.exists(src):
                print(f"Copying {csv_file} to {self.target_root}")
                shutil.copy(src, dst)
        
        super().parse(chunk_size)

    @property
    def keys(self) -> List[str]:
        """Identifier for image files."""
        return self._keys

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return 'PCXR'

    def _get_path(self, key: str) -> str:
        """Return file path for a given key."""
        return os.path.join(self.root, 'train' if self.is_train else 'test', key)

    def _get_meta_data(self, key: str) -> Dict:
        """Obtain meta data for a given key."""
        return self.meta_dict.get(key)  

    def _get_image(self, key: str) -> Image:
        """Load and process an image for a given key."""
        # Get image method needs to be overridden here, as ground truth is DICOM.
        ds = dcmread(os.path.join(self.img_dir, key), force=True)

        # Fix wrong metadata to prevent warning
        ds.BitsStored = 16

        try: 
            arr = ds.pixel_array
        except Exception as e:
            print(f"\n[ERROR] Skipping {key}: {e}")
            return None
        
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # Some images have a different mode
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            arr = 1.0 - arr

        img = Image.fromarray(np.uint8(arr * 255))
        img = img.convert('RGB')
        img = TRANSFORMS(img)
        return img

# pcxr_png
# --------------------------------------------------------------------------------------
class ParseCompositor:
    """Class for composing pcxr_png."""

    def __init__(
        self,
        target_root: str,
        pcxr_dicom_root: Optional[str] = None,
        transforms: Optional[Compose] = None,
        num_workers: int = 16,
        frontal_only: bool = True,
    ) -> None:
        """Initialize pcxr_png constructor."""
        self.pcxr_dicom_root = pcxr_dicom_root
        self.target_root = target_root
        self.transforms = transforms
        self.num_workers = num_workers
        self.frontal_only = frontal_only

    def _get_parser_objs(self) -> List[BaseParser]:
        """Instantiate parser objects."""
        ps = []

        if self.pcxr_dicom_root is not None:
            p = PCXRParser(
                root=self.pcxr_dicom_root,
                target_root=os.path.join(self.target_root, 'train'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
                is_train=True
            )
            ps.append(p)

            p = PCXRParser(
                root=self.pcxr_dicom_root,
                target_root=os.path.join(self.target_root, 'test'),
                transforms=self.transforms,
                num_workers=self.num_workers,
                frontal_only=self.frontal_only,
                is_train=False
            )
            ps.append(p)

        if len(ps) == 0:
            raise ValueError('No Datasets were specified for parsing.')

        print(
            '{} Datasets were specified for parsing: {}'.format(
                len(ps), ', '.join([r.name for r in ps])
            )
        )

        return ps

    def run(self) -> None:
        print('---------> Starting composition of pcxr_png <---------')
        ps = self._get_parser_objs()

        print('Target directory: {}'.format(self.target_root))
        print('{} workers are spawned.'.format(self.num_workers))
        os.makedirs(self.target_root, exist_ok=True)

        if self.frontal_only:
            print('The parser will only consider frontal scans.')
        else:
            print('The parser will consider frontal and lateral scans.')

        for p in ps:
            print('Parsing {:15s} with {:6d} samples.'.format(p.name, len(p)))
            p.parse()
            print('\nParsing {:15s} was successful.'.format(p.name))

            print('----------------------------------------------------')


if __name__ == '__main__':
    pcxr_png = ParseCompositor(
        target_root=cfg.pcxr_png_root,
        pcxr_dicom_root=cfg.pcxr_dicom_root,
        transforms=TRANSFORMS,
        num_workers=cfg.num_workers,
        frontal_only=cfg.frontal_only,
    )
    pcxr_png.run()
