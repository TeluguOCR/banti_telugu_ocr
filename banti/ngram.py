import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)
logi = logger.info
logd = logger.debug


class Ngram():
    def __init__(self, ngram_file):
        with open(ngram_file, 'rb') as fp:
            self._loaded = pickle.load(fp)

        self.n = len(self._loaded)

        # Start with a zero gram and add uni, bi, tri, etc.
        self.grams = [sum(self._loaded[0].values())]
        for i in range(self.n):
            self.grams.append(self._loaded[i])

    def __getitem__(self, glyphs):
        n = len(glyphs)
        dictionary = self.grams[n]
        for i in range(n):
            dictionary = dictionary[glyphs[i]]
        return dictionary

    def __call__(self, glyphs):
        if len(glyphs) == 0:
            return 0

        glyphs = glyphs[-self.n:]
        looked_up = '|'.join(glyphs)

        denom, numer = 0, 0
        try:
            denom = self[glyphs[:-1]]
            numer = self[glyphs]
        except KeyError:
            pass

        if denom == 0:
            ret = -6
        elif numer == 0:
            ret = -12
        else:
            ret = np.log(numer / denom)

        logd('|{}| :\t{}/{}\te^{:.3f}'.format(looked_up, numer, denom, ret))
        return ret