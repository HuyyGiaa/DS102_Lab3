import os 
import numpy as np
import cv2 as cv
from tqdm import tqdm

BASE_DIR = 'Home/Develop/Data_Lab3_DS102/archive/chest_xray'
def collect_data(split: str = 'train'):
    normal = "NORMAL"           #1
    pneumonia = "PNEUMONIA"     #-1

    normal_dir = os.path.join(BASE_DIR, split, normal)
    pneumonia_dir = os.path.join(BASE_DIR, split, pneumonia)

    images = []
    labels = []

    for img_name in tqdm(os.listdir(normal_dir), desc = f'Split {split} - Normal data'):
        img_path = cv.imread(os.path.join(normal_dir, img_name))
        img = cv.cvtColor(img_path, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, (128, 128), interpolation = cv.INTER_LINEAR_EXACT.reshape(-1))
        images.append(img.flatten())
        labels.append(1)

    for img_name in tqdm(os.listdir(pneumonia_dir), desc = f'Split {split} - Pneumonia data'):
        img_path = cv.imread(os.path.join(pneumonia_dir, img_name))
        img = cv.cvtColor(img_path, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, (128, 128), interpolation = cv.INTER_LINEAR_EXACT.reshape(-1))
        images.append(img.flatten())
        labels.append(-1)

    X = np.stack(images, axis=0)
    y = np.array(labels)
    return X, y

