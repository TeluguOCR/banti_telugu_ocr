#*-* encoding:utf-8 *-*
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
    @classmethod
    def fromSixPack(cls, str_entry):
        obj = cls()
        entries = str_entry.split()
        try:
            obj.ID = entries.pop(0)
            obj.x = int(entries.pop(0))
            obj.y = int(entries.pop(0))
            obj.wd = int(entries.pop(0))
            obj.ht = int(entries.pop(0))
            obj.baseline = int(entries.pop(0))
            obj.topline = int(entries.pop(0))
            obj.linenum = int(entries.pop(0))
            obj.wordnum = int(entries.pop(0))
            obj.sixpack = entries.pop(0)
        except IndexError:
            return None

        # Diff top & Diff bottom
        obj.dtop = obj.y - obj.topline
        obj.dbot = obj.y + obj.ht - obj.baseline
        obj.xht = obj.baseline - obj.topline

        # Process the 6packed string
        obj.pix = np.empty((obj.ht, obj.wd), dtype=np.uint8)
        for ipix in range(obj.ht * obj.wd):
            row, col, istr = ipix // obj.wd, ipix % obj.wd, ipix // 6
            obj.pix[row, col] = bool(
                (ord(obj.sixpack[istr]) - ord('0')) & (1 << (5 - (ipix % 6))))

        obj.img = im.fromarray(255 * (1 - obj.pix))
        return obj

    @classmethod
    def fromImg(cls, img, dtop, dbot):
        obj = cls()
        obj.img = img
        obj.dtop = dtop
        obj.dbot = dbot
        obj.wd, obj.ht = obj.img.size
        obj.xht = obj.ht + obj.dtop - obj.dbot
        obj.pix = np.array(img.convert('1').getdata(), 'uint8')
        obj.pix = 1 - (obj.pix.reshape((obj.ht, obj.wd)) / 255.)
        return obj

    @classmethod
    def fromImgDTBpairs(cls, img, dtopbot):
        """
        :type dtopbot: list of doubles like [(dt1, db1), (dt2, db2), ..., ]
        """
        obj = cls.fromImg(img, dtopbot[0][0], dtopbot[0][1])
        obj.dtopbot = dtopbot
        return obj

    def __str__(self):
        ret = '-'*(self.wd + 2) + '\n'

        for r in range(self.ht):
            ret += '|'
            for c in range(self.wd):
                ret += shade(self.pix[r, c])
            ret += '|\n'

        ret += '-'*(self.wd + 2) + '\n'

        return ret