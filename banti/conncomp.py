import scipy.ndimage.measurements as meas

class Component():
    def __init__(self, big_img, slice, index):
        self.index = index
        self.arr = big_img[slice] == index
        self.slice = slice

        self.y0, self.y1 = slice[0].start, slice[0].stop
        self.ht = self.y1 - self.y0
        self.x0, self.x1 = slice[1].start, slice[1].stop
        self.wd = self.x1 - self.x0

    def __lt__(self, other):
        overlap = max(0, min(self.x1 - other.x0, other.x1 - self.x0))

        if overlap / min(self.wd, other.wd) < .5:
            return self.x0 < other.x0
        else:
            return self.y0 + self.ht/2 < other.y0 + other.ht/2

    def __contains__(self, item):
        return item.x0 >= self.x0 and item.x1 <= self.x1 and \
               item.y0 >= self.y0 and item.y1 <= self.y1

    def small_str(self):
        return "Index:{} Range x: {}-{}({}) y:{}-{}({})\n".format(self.index,
            self.x0, self.x1, self.wd,
            self.y0, self.y1, self.ht)

    def __str__(self):
        return "Index:{} Range x: {}-{}({}) y:{}-{}({})\n".format(self.index,
            self.x0, self.x1, self.wd,
            self.y0, self.y1, self.ht) +\
               "\n".join([
                   "".join([" #"[c] for c in row])
                   for row in self.arr
               ])


def get_conn_comp(imgarr, sort=True):
    labelled_image, n_components = meas.label(imgarr)
    slices = meas.find_objects(labelled_image)
    components = []

    for islice, slice in enumerate(slices):
        components.append(Component(labelled_image, slice, islice+1))

    if sort:
        components = sorted(components)

    return components, labelled_image