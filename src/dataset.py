import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import IMG_SIZE

class MyDataset(Dataset):
    def __init__(self, images_filepaths, name2label, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform
        self.name2label = name2label

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imdecode(np.fromfile(image_filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Ошибка чтения файла: {image_filepath}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_name = os.path.normpath(image_filepath).split(os.sep)[-3]
        label = self.name2label[label_name]
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image, label

def collect_subclass_data(root_path):
    all_files = []
    subclass_labels_for_split = []
    subclass_id_counter = 0

    for class_name in tqdm(sorted(os.listdir(root_path)), desc="Processing classes"):
        class_path = os.path.join(root_path, class_name)
        if not os.path.isdir(class_path): continue

        for subclass_name in sorted(os.listdir(class_path)):
            subclass_path = os.path.join(class_path, subclass_name)
            if not os.path.isdir(subclass_path): continue

            images = glob(os.path.join(subclass_path, '*.jpg')) + \
                     glob(os.path.join(subclass_path, '*.png')) + \
                     glob(os.path.join(subclass_path, '*.jpeg'))

            if not images: continue

            images = sorted(images)
            all_files.extend(images)
            subclass_labels_for_split.extend([subclass_id_counter] * len(images))
            subclass_id_counter += 1

    return np.array(all_files), np.array(subclass_labels_for_split)

def get_train_transforms():
    return A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(
        min_height=IMG_SIZE,
        min_width=IMG_SIZE,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=1.0
    ),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=5,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=0.5
    ),

    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        A.CLAHE(p=0.1),
    ], p=0.85),

    A.OneOf([
        A.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.0, hue=0.0),
        A.ColorJitter(brightness=0.0, contrast=0.1, saturation=0.0, hue=0.0),
        A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.1, hue=0.0),
    ], p=1),

    A.OneOf([
        A.GaussianBlur(blur_limit=(3), sigma_limit=(0.1, 1.0), p=0.5),
        A.GaussNoise(std_range=(0.01, 0.02), noise_scale_factor=0.7, p=0.9),
    ], p=9),

    A.CoarseDropout(
        num_holes_range=(1, 16),
        hole_height_range=(0.02, 0.1),
        hole_width_range=(0.02, 0.1),
        fill=0,
        p=0.75
    ),

    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    A.ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(
        min_height=IMG_SIZE,
        min_width=IMG_SIZE,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=1.0
    ),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
    ])