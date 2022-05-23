import os
from os.path import join
import cv2
import numpy as np
from for_each import for_each

DIR = "C:/Users/AORUS/dataset"


def deal(path):
    img = cv2.imread(path)
    N, M = img.shape[:2]
    img = img[39 : N - 39, 39 : M - 39]
    cv2.imwrite(path, img)


if __name__ == "__main__":
    for_each(DIR, deal)

