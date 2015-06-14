import math
import numpy as np
from PIL import Image as im
import logging
logger = logging.getLogger(__name__)
logi = logger.info
logd = logger.debug


def shade(val):
    if val < 0.0: return '-'
    if val < .15: return ' '
    if val < .35: return '.'
    if val < .65: return 'o'
    if val < .85: return '0'
    if val <= 1.: return '#'
    return '+'


class BasicGlyph():
    """
    Basic glyph just contains an Image, its top, bottom, xht, ht, wd
    """
    def __init__(self, img_info=None):
        if len(img_info) == 3:
            self.init_from_img_dtop_dbot(*img_info)

        elif len(img_info) == 2:
            self.init_from_img_dtop_dbot_pairs(*img_info)

    def init_from_img_dtop_dbot(self, img, dtop, dbot):
        """
        :param img: Pillow Image
        :param int dtop: where the top line is going relative to top of image
        :param int dbot: relative bottom line
        :return: None
        """
        self.img = img
        self.dtop = dtop
        self.dbot = dbot
        self.wd, self.ht = self.img.size
        self.xht = self.ht + self.dtop - self.dbot
        self.pix = np.array(img.convert('1').getdata(), np.uint8)
        self.pix = 1 - (self.pix.reshape((self.ht, self.wd)) / 255.)

    def init_from_img_dtop_dbot_pairs(self, img, dtopbot_pairs):
        """
        :param dtopbot_pairs: list of doubles like [(dt1, db1), (dt2, db2), ..]
        :type dtopbot_pairs: list

        """
        self.init_from_img_dtop_dbot(img, *dtopbot_pairs[0])
        self.dtopbot_pairs = dtopbot_pairs

    def __str__(self):
        ret = '-' * (self.wd + 2) + '\n'

        for r in range(self.ht):
            ret += '|'
            for c in range(self.wd):
                ret += shade(self.pix[r, c])
            ret += '|\n'

        ret += '-' * (self.wd + 2) + '\n'
        ret += 'size:({}, {}) xht:{} dtop:{} dbot:{}'.format(
            self.wd, self.ht,
            round(self.xht, 1),
            round(self.dtop, 1),
            round(self.dbot, 1))

        return ret


class Glyph():
    def __init__(self, line_info=None):
        """

        :param line_info: x, y, ht, wd, etc.
        :type line_info: str or list
        """
        if type(line_info) is str:
            self.init_from_str(line_info)

        elif type(line_info) is list:
            self.init_from_list(line_info)

        else:
            self.init_from_list(['', 0, 0, 0, 0, 0, 0, 0, 0, None])

    def init_from_str(self, line_info):
        line_info = line_info.rstrip().split()
        for i, val in enumerate(line_info):
            if 0 < i < 9:
                line_info[i] = int(val)

        self.init_from_list(line_info)

    def init_from_list(self, box_list):
        self.text, self.x, self.y, self.wd, self.ht, \
        self.baseline, self.topline, \
        self.linenum, self.wordnum, \
        self.sixpack = box_list

        self.fix_x2_y2()
        self.fix_dtop_dbot_xht()
        self.pix_from_sixpack()

    def fix_dtop_dbot_xht(self):
        # Diff top & Diff bottom
        self.dtop = self.y - self.topline
        self.dbot = self.y + self.ht - self.baseline
        self.xht = self.baseline - self.topline

    # X2 and Y2 are set pythonically. i.e. they are not part of the image
    # They are just outside
    def fix_x2_y2(self):
        self.y2 = self.y + self.ht
        self.x2 = self.x + self.wd

    def fix_wh(self):
        self.wd = self.x2 - self.x
        self.ht = self.y2 - self.y

    def set_xy_wh(self, xywh):
        self.x, self.y, self.wd, self.ht = xywh
        self.fix_x2_y2()

    def set_xy_xy(self, xyxy):
        self.x, self.y, self.x2, self.y2 = xyxy
        self.fix_wh()

    def pix_from_sixpack(self):
        # Process the 6packed string
        pix = np.empty((self.ht, self.wd), dtype=np.uint8)

        if self.sixpack:
            for ipix in range(self.ht * self.wd):
                row, col, istr = ipix // self.wd, ipix % self.wd, ipix // 6
                pix[row, col] = bool((ord(self.sixpack[istr]) - ord('0'))
                                          & (1 << (5 - (ipix % 6))))

        self.set_pix(pix)

    def set_pix(self, pix):
        self.pix = np.array(pix, dtype=np.uint8)
        self.img = im.fromarray(255 * (1 - self.pix))

    def sixpack_from_pix(self):
        s = [ord('0') for i in range(math.ceil(self.ht * self.wd / 6))]
        for row in range(self.ht):
            for col in range(self.wd):
                ipix = row * self.wd + col
                isix = ipix // 6
                s[isix] += self.pix[row][col] << (5 - (ipix % 6));
        self.sixpack = ''.join(chr(i) for i in s)

    def get_pixel(self, row, col):
        return self.pix[row, col]

    def get_pixel_abs(self, abs_row, abs_col):
        if self.y <= abs_row < self.y2 and self.x <= abs_col < self.x2:
            return self.get_pixel(abs_row - self.y, abs_col - self.x)
        else:
            return 0


    def __add__(self, other):
        logd("Adding {} + {}".format(self, other))
        x1, y1, x2, y2 = min(self.x, other.x), min(self.y, other.y), \
                         max(self.x2, other.x2), max(self.y2, other.y2)
        summ = self.__class__()
        summ.set_xy_xy((x1, y1, x2, y2))
        summ.set_pix([[(self.get_pixel_abs(r, c) or other.get_pixel_abs(r, c))
              for c in range(x1, x2)] for r in range(y1, y2)])

        summ.text = self.text + other.text
        summ.baseline, summ.topline = self.baseline, self.topline
        summ.fix_dtop_dbot_xht()
        summ.linenum, summ.wordnum = self.linenum, self.wordnum

        return summ

    def set_text(self, txt, err=''):
        self.text = txt
        self.error = err
        return self

    def area(self):
        return self.wd * self.ht

    def __str__(self):
        meta = ' '.join(str(i) for i in (self.text,
                                         self.x, self.y, self.wd, self.ht,
                                         self.baseline, self.topline,
                                         self.linenum, self.wordnum))

        return BasicGlyph.__str__(self) + '\n' + meta


if __name__ == "__main__":
    from math import sin, pi, cos
    halfsize = 40
    size = 2*halfsize + 1
    picture = np.zeros((size, size))
    for t in range(-135, 135):
        x = round(halfsize + halfsize * cos(pi * t / 180))
        y = round(halfsize + halfsize * sin(pi * t / 180))
        picture[x][y] = 1

    zoomsz = 1 * halfsize
    b = Glyph(['O', 0, 0, size, size, 0, 0, 0, 0, None])
    b.set_pix(picture)
    c = Glyph()
    for t in range(0, 360, 15):
        x = round(zoomsz + zoomsz * cos(pi * t / 180))
        y = round(zoomsz + zoomsz * sin(pi * t / 180))
        b.set_xy_wh((x, y, size, size))
        c = c + b

    print(b)
    print(c)