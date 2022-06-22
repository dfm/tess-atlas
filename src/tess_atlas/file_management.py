import os
import shutil
from typing import Optional
import tarfile


def mkdir(base, name=None):
    if name:
        newpth = os.path.join(base, name)
        dirname = base if "." in name else newpth
    else:
        newpth = base
        dirname = base
    os.makedirs(dirname, exist_ok=True)
    return newpth


def shutil_logpath(path, names):
    print(f"Copying {path}\r", flush=True, end="\r")
    return []  # nothing will be ignored


def copy_tree(src, dst, verbose: Optional[bool] = True):
    ignore = shutil_logpath if verbose else None
    shutil.copytree(src, mkdir(dst), ignore=ignore, dirs_exist_ok=True)


def make_tarfile(output_filename: str, source_dir: str):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
