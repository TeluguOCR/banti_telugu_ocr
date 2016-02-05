import os
from PIL import Image as im
import numpy as np

##############################################################################
#                   OS Related Helpers
##############################################################################


def get_ext_changer(fname):
    base, ext = os.path.splitext(fname)
    def ext_changer(ext):
        return base + ext
    return ext_changer


def change_ext(fname, ext):
    name, _ = os.path.splitext(fname)
    return name + ext


def is_file_of_type(fname, ext):
    if ext == 'tif':
        checks = '.tif', '.tiff', '.TIF', '.TIFF'

    elif ext == 'box':
        checks = '.box', '.BOX'

    elif ext == 'pdf':
        checks = '.pdf', '.PDF'

    elif ext == 'dir':
        return os.path.isdir(fname)

    elif ext == 'image':
        try:
            im.open(fname)
            return True
        except OSError:
            return False

    else:
        raise ValueError('Unknown extension', ext)

    for e in checks:
        if fname.endswith(e):
            return True
    return False

##############################################################################
#                   Picture Related Helpers
##############################################################################

def img_to_bin_arr(img):
    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    return 1 - (pix.reshape((ht, wd)) / 255.)


def bin_arr_to_rgb_img(arr):
    return bin_arr_to_img(arr).convert("RGB")


def bin_arr_to_img(arr):
    return im.fromarray((255 * (1 - arr)).astype("uint8"))


def arr_to_ascii_art(pix):
    ht, wd = pix.shape
    ret = '-' * (wd + 2) + '\n'

    for r in range(ht):
        ret += '|'
        for c in range(wd):
            ret += shade(pix[r, c])
        ret += '|\n'

    ret += '-' * (wd + 2) + '\n'
    return ret


def shade(val):
    lookup = (0, '-'), (.15, ' '), (.35, '.'), (.65, 'o'), (.85, 'O'), (1+1e-7, '#')

    for value, ret in lookup:
        if val < value:
            return ret

    return '+'
