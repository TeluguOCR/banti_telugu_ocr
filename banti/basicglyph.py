import numpy as np


def shade(val):
    vals = 0, .15, .35, .65, .85
    rets = '- .oO#'

    for value, ret in zip(vals, rets):
        if val < value:
            return ret
    else:
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