import os
from os.path import join
import cv2
import numpy as np

path = "dataset"
border = np.array([4, 237, 253])


def deal(img: np.ndarray):
    N, M = img.shape[:2]
    return img[39 : N - 39, 39 : M - 39]


for sub in os.listdir(path):
    for file in os.listdir(join(path, sub)):
        cur_dir = join(path, sub, file)
        img: np.ndarray = cv2.imread(cur_dir, cv2.IMREAD_COLOR)
        cv2.imwrite(cur_dir, deal(img))
