# *-* coding:utf-8 *-*
import ast
import os
import sys
from math import ceil
from PIL import Image as im
import numpy as np


def tile_raster_images(images):
    n_images = images.shape[0]
    im_per_row = n_images // int(np.sqrt(n_images))
    im_per_col = ceil(float(n_images) / im_per_row)
    h, w = images.shape[1], images.shape[2]

    out_shape = (h + 1) * im_per_col - 1, (w + 1) * im_per_row - 1
    out_array = np.zeros(out_shape, dtype='uint8')

    for i in range(n_images):
        tile_row, tile_col = i // im_per_row, i % im_per_row
        out_array[
        tile_row * (h + 1): tile_row * (h + 1) + h,
        tile_col * (w + 1): tile_col * (w + 1) + w
        ] = 255 * (images[i])

    return out_array


def read_json_bz2(path2data):
    import bz2, json

    bz2_fp = bz2.BZ2File(path2data, 'r')
    data = np.array(json.loads(bz2_fp.read().decode('utf-8')))
    bz2_fp.close()
    return data

#######################################################################

def main():
    if len(sys.argv) < 4:
        print("Usage:python3 {0} images.bz2 labels.bz2 labelings.lbl"
              "\n\te.g:- python3 {0} num.images.bz2 num.labels.bz2 "
              "../labelings/numbers09.lbl"
              "\n"
              "Prints tiled images of all images of one class".format(sys.argv[0]))
        sys.exit(-1)

    data_file_name = sys.argv[1]
    labels_file_name = sys.argv[2]
    labelings_file_name = sys.argv[3]

    with open(labelings_file_name, 'r') as labels_fp:
        labellings = ast.literal_eval(labels_fp.read())
    reverse_labels = dict((v, k) for k, v in labellings.items())

    imgs = read_json_bz2(data_file_name)
    print('avg brightness = ', imgs.mean())
    labels = read_json_bz2(labels_file_name)
    dir_name = data_file_name.replace('.bz2', '/')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    namer = (dir_name + '{}.bmp').format

    for l in range(max(labels)):
        indices = labels == l
        name = namer(reverse_labels[l])

        imgs_l = imgs[indices]
        print("{}) Printing {} with {} images".format(l, name, imgs_l.shape[0]))
        im.fromarray(tile_raster_images(imgs_l)).save(name)

if __name__ == "__main__":
    main()