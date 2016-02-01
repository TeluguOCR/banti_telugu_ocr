from PIL import Image
from ..basicglyph import BasicGlyph


class Bunch(object):
    def __init__(self, adict):
        assert 'NMXHT' in adict
        assert 'NMTOP' in adict
        assert 'NMBOT' in adict
        assert 'NMXWD' in adict

        assert 'BUFLEFT' in adict
        assert 'BUFTOP' in adict
        assert 'BUFBOT' in adict
        assert 'SCALE_BY_TOP_BOTTOM_TOO' in adict
        self.__dict__.update(adict)
        self.NMHIT = self.NMTOP + self.NMXHT + self.NMBOT
        self.TOTWD = self.NMWID + self.BUFLEFT
        self.TOTHT = self.NMHIT + self.BUFTOP + self.BUFBOT


class Absolute:
    def __init__(self, params):
        self.params = Bunch(params)

    def __call__(self, glp):
        p = self.params

        # Find the amount to scale by
        scale = min(float(p.NMXHT) / glp.xht, float(p.NMWID) / glp.wd)
        if p.SCALE_BY_TOP_BOTTOM_TOO:
            if glp.dtop < 0:
                scale = min(scale, -float(p.NMTOP) / glp.dtop)
            if glp.dbot > 0:
                scale = min(scale, float(p.NMBOT) / glp.dbot)

        # Scale!
        new_wd = int(scale * glp.wd)
        new_ht = int(scale * glp.ht)
        scaled_img = glp.img.resize((new_wd, new_ht))

        # Find where the image should be moved
        #   (centered and adjusted to baseline)
        move2x = 0 if new_wd >= p.NMXWD else (p.NMXWD - new_wd) // 2
        scaldb = int(scale * glp.dbot)
        move2y = p.NMTOP + p.NMXHT - (new_ht - scaldb)

        img2 = Image.new("L", (p.TOTWD, p.TOTHT), "white")
        img2.paste(scaled_img, (p.BUFLEFT + move2x, p.BUFTOP + move2y))

        return BasicGlyph((img2, -p.BUFTOP-p.NMTOP, p.BUFBOT+p.NMBOT))