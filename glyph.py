# *-* encoding:utf-8 *-*
import math
import numpy as np
from PIL import Image as im


def shade(val):
    if val < 0.0:  return '-'
    if val < .15:  return ' '
    if val < .35:  return '.'
    if val < .65:  return 'o'
    if val < .85:  return '0'
    if val <= 1.:  return '#'
    return '+'


class Glyph(object):
    def __init__(self,
                 line_info=None,
                 img_info=None,
                 ):
        """

        :param str line_info:
        :param list img_info:
        :param list list_info:
        """
        if line_info:
            if type(line_info) is str:
                self.init_from_list(line_info.rstrip().split())

            elif type(line_info) is list:
                self.init_from_list(line_info)

        elif img_info:
            if len(img_info) == 2:
                self.init_from_img_dtop_dbot(*img_info)

            elif len(img_info) == 3:
                self.init_from_img_dtop_dbot_pairs(*img_info)

        else:
            self.init_from_list(['', 0, 0, 0, 0, 0, 0, 0, 0, ''])

    def init_from_list(self, box_list):
        self.text, self.x, self.y, self.wd, self.ht, \
        self.baseline, self.topline, \
        self.linenum, self.wordnum, \
        self.sixpack = box_list

        self.fix_all()

    def init_from_img_dtop_dbot(self, img, dtop, dbot):
        """
        Initialized this way, the img does not have location info!
        So you can not add images!
        :param img: Pillow Image
        :param int dtop:
        :param int dbot:
        :return:
        """
        self.img = img
        self.dtop = dtop
        self.dbot = dbot
        self.wd, self.ht = self.img.size
        self.xht = self.ht + self.dtop - self.dbot
        self.pix = np.array(img.convert('1').getdata(), np.uint8)
        self.pix = 1 - (self.pix.reshape((self.ht, self.wd)) / 255.)
        return self

    def init_from_img_dtop_dbot_pairs(self, img, dtopbot_pairs):
        """
        :type dtopbot_pairs: list of doubles like [(dt1, db1), (dt2, db2), ..., ]
        """
        self.init_from_img_dtop_dbot(img, *dtopbot_pairs[0])
        self.dtopbot_pairs = dtopbot_pairs

    def pix_as_list(self):
        return [[self.get_pixel(row, col) for col in range(self.ht)] \
                for row in range(self.wd)]

    def fix_all(self):
        self.fix_x2_y2()
        self.pix_from_sixpack()
        self.fix_dtop_dbot_xht()

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
        self.pix = np.empty((self.ht, self.wd), dtype=np.uint8)
        for ipix in range(self.ht * self.wd):
            row, col, istr = ipix // self.wd, ipix % self.wd, ipix // 6
            self.pix[row, col] = bool(
                (ord(self.sixpack[istr]) - ord('0')) & (1 << (5 - (ipix % 6))))

        self.img = im.fromarray(255 * (1 - self.pix))

    def sixpack_from_pix(self):
        s = [ord('0') for i in range(math.ceil(self.ht * self.wd / 6))]
        for row in range(self.ht):
            for col in range(self.wd):
                ipix = row * self.wd + col
                isix = ipix // 6
                s[isix] += self.pix[row][col] << (5 - (ipix % 6));
        self.sixpack = ''.join(chr(i) for i in s)

    def __str__(self):
        ret = '-' * (self.wd + 2) + '\n'

        for r in range(self.ht):
            ret += '|'
            for c in range(self.wd):
                ret += shade(self.pix[r, c])
            ret += '|\n'

        ret += '-' * (self.wd + 2) + '\n'

        meta = ' '.join(str(i) for i in (self.text, self.x, self.y,
                                         self.wd, self.ht, self.baseline,
                                         self.topline,
                                         self.linenum, self.wordnum,
                                         self.pix)) + '\n'

        return ret + meta

    def get_pixel(self, row, col):
        return self.pix[row, col]

    def get_pixel_abs(self, abs_row, abs_col):
        if self.y <= abs_row < self.y2 and self.x <= abs_col < self.x2:
            return self.get_pixel(abs_row - self.y, abs_col - self.x)
        else:
            return 0


    def __add__(self, other):
        x1, x2, y1, y2 = min(self.x, other.x), min(self.y, other.y), \
                         max(self.x2, other.x2), max(self.y2, other.y2)
        summ = Glyph()
        summ.pix = np.array(
            [[(self.get_pixel_abs(r, c) or other.get_pixel_abs(r, c))
              for c in range(x1, x2)] for r in range(y1, y2)],
            dtype=np.uint8)
        summ.img = im.fromarray(255 * (1 - summ.pix))

        summ.text = self.text + other.text
        summ.set_xy_xy((x1, y1, x2, y2))
        summ.baseline, summ.topline = self.baseline, self.topline
        summ.fix_dtop_dbot_xht()
        summ.linenum, summ.wordnum = self.linenum, self.wordnum
        summ.sixpack = None

        return summ

    def set_text(self, txt, err=''):
        self.text = txt
        self.error = err
        return self

    def area(self):
        return self.wd * self.ht


def FindBigSmallBoxes(boxes):
    imax = 0
    imin = 0
    for i in range(1, len(boxes)):
        if boxes[imax].area() < boxes[i].area():
            imax = i
        if boxes[imin].area() >= boxes[i].area():
            imin = i
    return (imax, imin)


def GetBoxesForLine(file_name):
    with open(file_name) as box_file:
        i, l = 0, []
        for line in box_file:
            box = Glyph(line_info=line)
            if box.linenum != i:
                yield l
                i, l = box.linenum, []
            l.append(box)
        yield l
