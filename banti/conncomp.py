import scipy.ndimage.measurements as meas
from .helpers import arr_to_ascii_art


class Component():
    def __init__(self, big_img, slice, index):
        self.index = index
        self.pix = big_img[slice] == index
        self.slice = slice

        self.y, self.y2 = slice[0].start, slice[0].stop
        self.ht = self.y2 - self.y
        self.x, self.x2 = slice[1].start, slice[1].stop
        self.wd = self.x2 - self.x

    def __lt__(self, other):
        overlap = max(0, min(self.x2 - other.x, other.x2 - self.x) - 1)

        if overlap / min(self.wd, other.wd) < .5:
            return self.x < other.x
        else:
            return self.y + self.ht/2 < other.y + other.ht/2

    def __contains__(self, item):
        if isinstance(item, Component):
            return item.x >= self.x and item.x2 <= self.x2 and \
                   item.y >= self.y and item.y2 <= self.y2
        else:
            raise NotImplementedError("Type of item is unknown: " + type(item))

    def has_center_of(self, other):
        return self.x <= (other.x + other.x2)/2 <= self.x2 and \
               self.y <= (other.y + other.y2)/2 <= self.y2

    def small_str(self):
        return "Index:{} Range x: {}-{}({}) y:{}-{}({})\n".format(self.index,
            self.x, self.x2, self.wd,
            self.y, self.y2, self.ht)

    def __str__(self):
        return self.small_str() + "\n" + arr_to_ascii_art(self.pix)


def get_conn_comp(imgarr, sort=True):
    labelled_image, n_components = meas.label(imgarr)
    slices = meas.find_objects(labelled_image)
    components = []

    for islice, slaiss in enumerate(slices):
        components.append(Component(labelled_image, slaiss, islice+1))

    if sort:
        components = sorted(components)

    return components, labelled_image