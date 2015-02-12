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
        self.tricount = None

    def set_trigram(self, pklfile):
        with open(pklfile, 'rb') as fp:
            self.tricount = pickle.load(fp)
        print(type(self.tricount))


    def __call__(self, chars):
        """ Look up the trigram matrix to get probability of the a sequence of
            one, two or three given glyphs
            TODO : Do better smoothing based on unigram and bigram frequencies
        """
        if len(chars) < 2:
            return 0

        if self.tricount is None:
            raise ValueError("Trigram not set")

        s = chars[-3] if len(chars) > 2 else ' '
        try:
            val = self.tricount[s][chars[-2]][chars[-1]]
            if val:
                print('Found: {} : {}'.format(''.join(chars[-3:]), val))
        except KeyError:
            val = 0
            print("Not found {}".format(''.join(chars[-3:])))
            # raise KeyError("Not found {}".format(''.join(chars[-3:])))

        return np.log(1e-6 + val)

priorer = _Prior()


class Path():
    """ Class implements one instance of a probable path
        chars       : the characters/glyphs making up the path
        posterior   : log posterior probability (sum of liklihood and gramprior)
        liklihood   : log liklihood probability
        gramprior   : log prior probability
    """

    def __init__(self, arg=None):
        if arg is None:
            self.chars = []
            self.posterior = 0
            self.liklihood = 0
            self.gramprior = 0

        elif isinstance(arg, Path):
            self.chars = arg.chars[:]
            self.posterior = arg.posterior
            self.liklihood = arg.liklihood
            self.gramprior = arg.gramprior

        else:
            raise TypeError("Got invalid arg {} of type {}".format(
                arg, type(arg)))

    def add_next_char(self, char, lik=0):
        self.chars.append(char)
        self.update_posterior(lik)

    def update_posterior(self, lik):
        self.liklihood += lik
        self.gramprior += priorer(self.chars)
        self.posterior = self.liklihood + self.gramprior

    def beget(self, char, lik):
        child = Path(self)
        child.add_next_char(char, lik)
        return child

    def __str__(self):
        return '{} : LIK{:3.2f} + PRI{:3.2f} = PST{:3.2f}'.format(
            ''.join(self.chars),
            self.liklihood,
            self.gramprior,
            self.posterior,
        )

    def text(self):
        return ''.join(self.chars)


class Paths():
    """ Class implements a set of probable OCR paths
        order: maximum number of paths you want stored
        paths: the actual ocr_paths
    """

    def __init__(self, order):
        self.order = order
        self.paths = [Path()]

    def update(self, liklies):
        """ Given the set of probable characters for the next glyphs,
            creates more child paths and keeps only the best ones.
        """
        tmp_paths = [path.beget(ch, lk) for path in self.paths
                     for ch, lk in liklies]
        tmp_paths = sorted(tmp_paths, key=lambda p: p.posterior, reverse=True)
        self.paths = tmp_paths[:self.order]

    def simple_update(self, char, sort=False):
        """ Just adds the same character to all the paths (usually a space)"""
        for path in self.paths:
            path.add_next_char(char)
        if sort:
            self.paths = sorted(self.paths, key=lambda p: p.posterior,
                                reverse=True)

    def print_top(self, n=3):
        for i in range(min(len(self.paths), n)):
            print(str(self.paths[i]))
        print('-'*n)