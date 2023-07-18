import os
import shutil
import tarfile
from typing import Optional

# CONSTANT FILENAMES
SAMPLES_FNAME = "samples.csv"
INFERENCE_DATA_FNAME = "inference_data.netcdf"
TOI_DIR = "toi_{toi}_files"
TIC_CSV = "tic_data.csv"
PROFILING_CSV = "profiling.csv"
LC_DATA_FNAME = "lightkurve_lc.fits"


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
    try:
        shutil.copytree(src, mkdir(dst), ignore=ignore, dirs_exist_ok=True)
    except shutil.Error as e:
        print(f"Error copying {src}->{dst}: {e}")


def make_tarfile(output_filename: str, source_dir: str):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def read_last_n_lines(file_path, n):
    with open(file_path, "rb") as file:
        file.seek(0, 2)  # Move the file pointer to the end of the file
        file_size = file.tell()  # Get the current position (file size)

        lines = []
        line_count = 0
        for i in range(file_size - 1, 0, -1):
            file.seek(i)
            char = file.read(1)
            if char == b"\n":
                lines.append(file.readline().decode().strip())
                line_count += 1
                if line_count == n:
                    break
        lines.reverse()  # Reverse the lines to get them in the correct order
        return "\n".join(lines)


def get_filesize(path):
    """Get the size of a file in Mb"""
    bytes = os.path.getsize(path)
    return bytes / 1e6
