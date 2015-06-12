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

if __name__ == '__main__':
    ngram_file = "library/mega.123.pkl"

    txt = [['రా', 'మ', 'జో', 'గి', 'మ', 'ం', 'దు', ' ', 'కొ', 'న', 'రే',
            ' ', 'ఓ', 'జ', 'ను', 'లా', 'ర', ' ', 'రా', '.', '.'],
           ['అ', 'ను', '✓', 'ప', 'ల', '్ల', 'వి', ':'],
           ['రా', 'మ', 'జో', 'గి', 'మ', 'ం', 'దు', ' ', 'కొ', 'ని', ' ',
            'ే', 'ప', '్ర', 'మ', 'తో', ' ', 'భు', 'జి', 'యి', 'ం', 'చు', 'డ', 'న', '్న'],
           ['కా', 'మ', 'కో', '్ర', 'ధ', 'లో', 'భ', 'మో', '✓', 'హ', 'ఘ', 'న', 'మె', 'ై', 'న', ' ',
            'రో', 'గా', 'ల', 'కు', 'మ', 'ం', 'దు', 'రా', '.', '.'],
           ['చ', 'ర', 'ణ', 'ము', '(', 'లు', ')', ':'],
           ['కా', 'టు', 'క', 'కొ', 'ం', 'డ', 'ల', 'వ', 'ం', 'టి', ' ',
            'క', 'ర', '్మ', 'ము', 'లె', 'డ', 'బా', 'ే', 'ప', ' ', 'మ', 'ం', 'దు'],
           ['సా', 'టి', 'లే', 'ని', ' ', 'జ', 'గ', 'ము', 'న', 'ం', 'దు', ' ',
            'సా', '్వ', 'మి', 'రా', 'మ', 'జో', 'గి', 'మ', 'ం', 'దు', 'రా', '.', '.'],
           ['వా', 'దు', 'కు', ' ', 'చె', 'ి', 'ప', '్ప', 'న', 'గా', 'ని', ' ',
            'వా', 'రి', 'పా', '✓', 'ప', 'ము', 'లు', 'గొ', 'టి', '్ట'],
           ['ము', 'ద', 'ము', 'తో', 'నే', ' ', 'మో', 'క్ష', 'మి', 'చే', '్చ', ' ',
            'ము', 'దు', '్ద', ' ', 'రా', 'మ', 'జో', 'గి', 'మ', 'ం', 'దు', 'రా', '.', '.'],
           ['ము', 'ద', 'ము', 'తో', ' ', 'భ', 'దా', '్ర', 'ది', '్ర', 'య', 'ం', 'దు', ' ',
            'ము', 'కి', '్త', 'ని', ' ', 'పొ', 'ం', 'ది', 'ం', 'చే', 'మ', 'ం', 'దు'],
           ['స', 'ద', 'యు', 'డె', 'ై', 'న', ' ', 'రా', 'మ', 'దా', '✓', 'సు', ' ',
            'ము', 'ద', 'ము', 'తో', ' ', 'ే', 'స', 'వి', 'ం', 'చే', 'మ', 'ం', 'దు', 'రా']]

    ngram = Ngram(ngram_file)
    for text in txt:
        prior = 0
        for i in range(len(text)):
            prior += ngram(text[:i])

        print(prior)