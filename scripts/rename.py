import os
from os.path import join

from CONFIG import TRANSLATIONS

SRC = "加框后的JPEG图"
DST = "../dataset"
USE_SUBDIR = True


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    for big in os.listdir(SRC):
        for small in os.listdir(join(SRC, big)):
            cur = TRANSLATIONS[small]
            if USE_SUBDIR:
                os.makedirs(join(DST, cur), exist_ok=True)
            for i, file in enumerate(os.listdir(join(SRC, big, small))):
                # name = f"{cur}.{i}.jpg"
                name = f"{i}.jpg"
                if USE_SUBDIR:
                    os.rename(join(SRC, big, small, file), join(DST, cur, name))
                else:
                    os.rename(join(SRC, big, small, file), join(DST, name))
