import numpy as np
from math import ceil
from scipy.ndimage.interpolation import zoom


def normalize(img, make_white):
    maxx, minn = img.max(), img.min()
    img -= minn
    img /= maxx - minn
    if make_white and np.mean(img) < .5:
        img = 1 - img
    return img


def tile_raster_images(images,
          zm=1,
          margin_width=1,
          margin_color=.1,
          make_white=False,
          global_normalize=False):
    n_images = images.shape[0]
    w = n_images // int(np.sqrt(n_images))
    h = ceil(float(n_images) / w)

    if global_normalize:
        images = normalize(images, make_white)
    else:
        images = [normalize(img, make_white) for img in images]

    if zm != 1:
        images = zoom(images, zoom=(1, zm, zm), order=0)

    pad_axes = (0, h*w - n_images), (0, 1), (0, 1)
    pad_width = (margin_width * np.array(pad_axes)).tolist()
    pad_fill = (margin_color * np.array(pad_axes)).tolist()
    images = np.pad(images, pad_width, 'constant', constant_values=pad_fill)

    t2 = np.vstack([np.hstack([images[i * w + j] for j in range(w)])
                    for i in range(h)])
    t2 = t2[:-margin_width, :-margin_width]
    t2 = (255 * t2).astype("uint8")

    return t2