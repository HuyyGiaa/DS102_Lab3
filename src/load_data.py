import os 
import numpy as np
import cv2 as cv
from tqdm import tqdm
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR / '..' / 'archive' / 'chest_xray'
BASE_DIR = str(BASE_DIR)

def collect_data(split: str = 'train'):
    normal = "NORMAL"           #-1
    pneumonia = "PNEUMONIA"     #1

    normal_dir = os.path.join(BASE_DIR, split, normal)
    pneumonia_dir = os.path.join(BASE_DIR, split, pneumonia)

    images = []
    labels = []

    for img_name in tqdm(os.listdir(normal_dir), desc = f'Split {split} - Normal data'):
        img_path = cv.imread(os.path.join(normal_dir, img_name))
        img = cv.cvtColor(img_path, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, (128, 128), interpolation = cv.INTER_LINEAR_EXACT)
        images.append(img.flatten() / 255.0)
        labels.append(-1)

    for img_name in tqdm(os.listdir(pneumonia_dir), desc = f'Split {split} - Pneumonia data'):
        img_path = cv.imread(os.path.join(pneumonia_dir, img_name))
        img = cv.cvtColor(img_path, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, (128, 128), interpolation = cv.INTER_LINEAR_EXACT)
        images.append(img.flatten() / 255.0)
        labels.append(1)

    X = np.stack(images, axis=0)
    y = np.array(labels)
    return X, y

