from typing import List

from PIL import Image


def vertical_image_concat(filenames: List[str], outfilename: str):
    """Concatenate images vertically"""
    images = [Image.open(fn) for fn in filenames]
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGB", (total_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    new_im.save(outfilename)
