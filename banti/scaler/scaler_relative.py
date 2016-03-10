from PIL import Image
from ..basicglyph import BasicGlyph


class Bunch(object):
    def __init__(self, adict):
        assert 'HT_MARGIN' in adict
        assert 'WD_MARGIN' in adict
        assert 'WIDTH' in adict
        assert 'HEIGHT' in adict
        assert 'XHEIGHT' in adict
        self.__dict__.update(adict)
        self.TOTWD = self.WIDTH + 2 * self.WD_MARGIN
        self.TOTHT = self.HEIGHT + 2 * self.HT_MARGIN


class Relative():
    def __init__(self, params):
        self.params = Bunch(params)

    def __call__(self, glp):
        p = self.params

        # Find the amount to scale by
        rel_sizes = (float(glp.wd)/p.WIDTH,
                     float(glp.ht)/p.HEIGHT,
                     float(glp.xht)/p.XHEIGHT)
        scale = 1 / max(rel_sizes)

        # Scale!
        new_wd = int(scale * glp.wd)
        new_ht = int(scale * glp.ht)
        if (new_wd > 0) and (new_ht > 0):
            scaled_img = glp.img.resize((new_wd, new_ht))
        else:
            scaled_img = glp.img
            scale = 1
            new_wd = glp.wd
            new_ht = glp.ht

        # Place the scaled image in correct location
        move2x = p.WD_MARGIN + (p.WIDTH - new_wd)//2
        move2y = p.HT_MARGIN + (p.HEIGHT - new_ht)//2

        img2 = Image.new("1", (p.TOTWD, p.TOTHT), "white")
        img2.paste(scaled_img, (move2x, move2y))

        # Update Image
        scalef = float(new_ht)/glp.ht
        new_dtop = glp.dtop * scalef - move2y
        new_dbot = glp.dbot * scalef - move2y + p.TOTHT - new_ht

        return BasicGlyph((img2, new_dtop, new_dbot))
