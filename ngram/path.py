#! /usr/bin/env python3
"""
    Project : Telugu OCR
    Author  : Rakeshvara Rao
    License : GNU GPL 3.0

    This module implements the n-gram model for Telugu OCR
"""
import pickle
import numpy as np
from collections import defaultdict, Counter


def defaultdict_counter():
    return defaultdict(Counter)


class _Prior():
    def __init__(self):
        self.uni, self.bi, self.tri = None, None, None

    def set_trigram(self, pklfile):
        with open(pklfile, 'rb') as fp:
            self.uni, self.bi, self.tri = pickle.load(fp)

    def __call__(self, glyphs):
        """ Look up the trigram matrix to get probability of the a sequence of
            one, two or three given glyphs
            TODO : Do better smoothing based on unigram and bigram frequencies
        """
        if self.tri is None:
            raise ValueError("Trigram not set")

        if len(glyphs) == 0:
            return 0

        elif len(glyphs) == 1:
            try:
                num, denom = self.bi[" "][glyphs[0]], self.uni[" "]
                print('Y|{}| : {}/{}'.format('|'.join(glyphs).replace(" ", "_"), num, denom))
            except KeyError:
                num, denom = 0, 1
                print("X|{}|".format('|'.join(glyphs).replace(" ", "_")))

        else:
            a = glyphs[-3] if len(glyphs) > 2 else ' '
            try:
                num, denom = self.tri[a][glyphs[-2]][glyphs[-1]], self.bi[glyphs[-2]][glyphs[-1]]
                print('Y|{}| : {}/{}'.format('|'.join(glyphs[-3:]).replace(" ", "_"), num, denom))

            except KeyError:
                num, denom = 0, 1
                print("X|{}|".format('|'.join(glyphs[-3:]).replace(" ", "_")))

        return np.log(1e-6 + num/denom)

priorer = _Prior()


class Path():
    """ Class implements one instance of a probable path
        glyphs      : the glyphs making up the path
        posterior   : log posterior probability (sum of liklihood and gramprior)
        liklihood   : log liklihood probability
        gramprior   : log prior probability
    """

    def __init__(self, arg=None):
        if arg is None:
            self.glyphs = []
            self.posterior = 0
            self.liklihood = 0
            self.gramprior = 0

        elif isinstance(arg, Path):
            self.glyphs = arg.glyphs[:]
            self.posterior = arg.posterior
            self.liklihood = arg.liklihood
            self.gramprior = arg.gramprior

        else:
            raise TypeError("Got invalid arg {} of type {}".format(
                arg, type(arg)))

    def add_next_glyph(self, glyph, lik=0):
        self.glyphs.append(glyph)
        self.update_posterior(lik)

    def update_posterior(self, lik):
        self.liklihood += lik
        self.gramprior += priorer(self.glyphs)
        self.posterior = self.liklihood + self.gramprior

    def beget(self, glyph, lik):
        child = Path(self)
        child.add_next_glyph(glyph, lik)
        return child

    def __str__(self):
        return '{} : L{:.2f}+R{:.2f}=T{:.2f}'.format(
            ''.join(self.glyphs),
            self.liklihood,
            self.gramprior,
            self.posterior,
        )

    def text(self):
        return ''.join(self.glyphs)


class Paths():
    """ Class implements a set of probable OCR paths
        order: maximum number of paths you want stored
        paths: the actual ocr_paths
    """

    def __init__(self):
        self.paths = [Path()]

    def update(self, liklies):
        """ Given the set of probable candidates for the next glyph,
            creates more child paths and keeps only the best ones.
        """
        tmp_paths, n = [], 0
        for glp, lik in liklies:
            n += 1
            best = -np.inf
            for path in self.paths:
                child = path.beget(glp, lik)
                if child.posterior > best:
                    best = child.posterior
                    bestchild = child
            tmp_paths.append(bestchild)
            #print("{} {:.1e} {}".format(glp, lik, bestchild))

        self.paths = tmp_paths
        print("Recieved {} likelies, created {} paths".format(
            n, len(self.paths)))
        for p in sorted(self.paths, key=lambda x:x.posterior, reverse=True):
            print(p)

    def simple_update(self, glyph, sort=False):
        """ Just adds the same glyph to all the paths (usually a space)"""
        for path in self.paths:
            path.add_next_glyph(glyph)
        if sort:
            self.paths = sorted(self.paths, key=lambda p: p.posterior,
                                reverse=True)

    def print_top(self, n=3):
        print("Top", n, "candidates")
        for i in range(min(len(self.paths), n)):
            print(str(self.paths[i]))
