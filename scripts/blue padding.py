import cv2
import numpy as np
import glob
import os
from os.path import join, split


SRC = "C:/Users/AORUS/dataset"
DST = "C:/Users/AORUS/dataset test"
for name in glob.glob(join(SRC, "*/*")):
    img = cv2.imread(name)
    path, name = split(name)
    ret = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(204, 23, 23))
    cv2.imwrite(join(DST, f"{split(path)[1]}.{name}"), ret)
