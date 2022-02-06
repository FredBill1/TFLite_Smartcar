import glob
import os
import multiprocessing
import sys
from os.path import join, split, splitext
from typing import Tuple


def for_each(DIR: str, func: function, start: int = 0, batch_size: int = 2048) -> None:
    glb = glob.glob(join(DIR, "*/*"))
    N = len(glb)
    M = os.cpu_count()

    sys.stdout.write(f"{func.__name__}, {N} files\n")
    with multiprocessing.Pool(processes=M) as pool:
        for i in range(start, N, batch_size):
            r = min(i + batch_size, N)
            sys.stdout.write(f"\r{i}~{r}")
            pool.map(func, (glb[j] for j in range(i, r)), chunksize=(r - i) // M)
    sys.stdout.write("\n")


def parse_path(path: str) -> Tuple[str, str, str]:
    Dir, name = split(path)
    name, ext = splitext(name)
    return Dir, name, ext


__all__ = ["parse_path", "for_each"]
