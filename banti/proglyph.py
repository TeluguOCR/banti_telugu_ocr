import logging
import numpy as np

from .glyph import Glyph

logger = logging.getLogger(__name__)
logi = logger.info
logd = logger.debug


def do_combine(self, other):
    """

    :param ProGlyph self:
    :param ProGlyph other:
0    :return:
    """
    score, yell = 0, ""
    s, t = self.best_char, other.best_char
    ss, st = self.strength(), other.strength()

    suspects = ['ఏ', '-', '"', "'", '.', 'ై']
    ghpsshh = 'ఘపసషహఎ'
    heads = ['ి', 'ీ', 'ె', 'ే', '✓', '్']

    def is_sharer(c):
        return c[0] in heads or c is 'ై' or c in 'ఖఙజఞణ'

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


class ProGlyph(Glyph):
    """Class used to process a space seperated line and store the probable
    characters and the respective liklihoods for one glyph.
    """
    scaler = lambda *_: None
    classifier = lambda *_: (("", 0),)
    ngram = ()

    def __init__(self, info=None):
        super().__init__(info)
        if info:
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