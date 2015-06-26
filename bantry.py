import numpy as np
import logging
from glyph import Glyph

logger = logging.getLogger(__name__)
logi = logger.info
logd = logger.debug


def do_combine(self, other):
    """

    :param Bantry self:
    :param Bantry other:
0    :return:
    """
    score, yell = 0, ""
    s, t = self.best_char, other.best_char
    ss, st = self.strength(), other.strength()

    suspects = ['ఏ', '-', '"', "'", '.', 'ై',]
    ghpsshh = 'ఘపసషహఎ'
    heads = ['ి', 'ీ', 'ె', 'ే', '✓', '్']
    def is_sharer(c):
        return c[0] in heads or \
               c is 'ై' or \
               c in 'ఖఙజఞణ'

    lowprob = np.log(.95)
    vlprob = np.log(.70)
    if other is Space:
        return False

    # Strong rules resulting in immediate failure
    if s in heads and t[0] not in ghpsshh:
        score += 5
        yell += '*PSHG'

    # Weak rules, that help each other
    if self.area < self.xarea/4:
        score += 1
        yell += '+AREAS'
    if other.area < self.xarea/4:
        score += 1
        yell += '+AREAO'

    overlap = self.overlap(other)
    if overlap > .75:
        score += 1
        yell += '+OVLP:{:.0f}'.format(100*overlap)

    if t in suspects:
        score += 1
        yell += '+SUSPO'

    if s in suspects:
        score += 1
        yell += '+SUSPS'

    if not is_sharer(s) and not is_sharer(t) and overlap > .5:
        score += 2
        yell += '#OLNS'

    try:
        self.ngram[self.best_char, other.best_char]
    except KeyError:
        yell += '+DICT'
        score += 1

    if ss < vlprob:
        score += 1
        yell += '+VLPS{:.0f}'.format(100*np.exp(ss))

    if st < vlprob:
        score += 1
        yell += '+VLPO{:.0f}'.format(100*np.exp(st))

    if (ss < lowprob) and (st < lowprob):
        score += 1
        yell += '+LOWP{:.0f}&{:.0f}'.format(100*np.exp(ss), 100*np.exp(st))

    if score > 1:
        combined_area = self.combined_area(other)
        if combined_area < 3 * self.xarea:
            logi("Combining 'cos: " + yell)
            return score > 1
        else:
            logi("Combined Area too big: {}>3*{}".format(combined_area, self.xarea))


class Bantry(Glyph):
    """Class used to process a space seperated line and store the probable
    characters and the respective liklihoods for one glyph.
    """
    scaler = lambda *_: None
    classifier = lambda *_: (("", 0),)
    ngram = ()

    def __init__(self, line_info=None):
        super().__init__(line_info)
        if line_info:
            self.scaled = self.scaler(self)
            self.likelies = self.classifier(self.scaled)
            logd("Initialized\n{}".format(self))

    @property
    def best_char(self):
        return max(self.likelies, key=lambda x: x[1])[0]

    def strength(self):
        return max(self.likelies, key=lambda x: x[1])[1]

    @property
    def strlikelies(self):
        return " ".join("{}{:.4f}".format(char, np.exp(lik)) for char, lik in self.likelies)

    def __str__(self):
        return super().__str__() + "\n" + self.strlikelies

    def combine(self, other):
        logd("Checking to combine\n{}\n{}".format(self, other))

        if do_combine(self, other):
            combined = self + other
            combined.scaled = self.scaler(combined)
            combined.likelies = self.classifier(combined.scaled)
            if logger.isEnabledFor(logging.DEBUG):
                logi("Combining\n{}".format(combined))
            else:
                logi("Combining\n{}\n{}\n{}".format(self, other, combined))

            return True, combined
        else:
            return False, None


class Space():
    likelies = [(" ", 0)]
    strlikelies = " : 0"
    best_char = " "
    strength = lambda: 0
    scaled = "---\n| |\n---"

    @classmethod
    def combine(cls, other):
        return False, None

    @classmethod
    def __str__(cls):
        return "_"


class BantryFile():
    def __init__(self, name):
        in_file = open(name)
        self.file_bantries = []

        iword, iline = 0, 0
        line_bantries = []

        for line in in_file:
            e = Bantry(line)
            if e.linenum == iline:
                if e.wordnum > iword:
                    iword = e.wordnum
                    line_bantries.append(Space)
                line_bantries.append(e)

            elif e.linenum > iline:
                self.file_bantries.append(line_bantries)
                iword = 0
                iline += 1
                while iline < e.linenum:
                    self.file_bantries.append([])
                    iline += 1
                line_bantries = [e]

            else:
                raise ValueError("Line number can not go down.")

        self.file_bantries.append(line_bantries)
        self.num_lines = self.file_bantries[-1][-1].linenum

        self.text = ""
        for bantries_inline in self.file_bantries:
            for bantree in bantries_inline:
                self.text += bantree.best_char
            self.text += "\n"

        in_file.close()

    def get_line_bantires(self, i):
        return self.file_bantries[i]

if __name__ == "__main__":
    import sys
    from scaler import ScalerFactory

    banti_file_name = sys.argv[1] if len(sys.argv) > 1 else "sample_images/praasa.box"
    scaler_prms_file = sys.argv[2] if len(sys.argv) > 2 else "scalings/relative48.scl"

    Bantry.scaler = ScalerFactory(scaler_prms_file)
    bf = BantryFile(banti_file_name)

    for linenum in range(bf.num_lines):
        print('*' * 60)
        line_bantries = bf.get_line_bantires(linenum)
        for bantry in line_bantries:
            print(bantry.scaled)