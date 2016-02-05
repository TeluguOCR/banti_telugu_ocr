#!/usr/bin/env python3
"""
    Project : Telugu OCR
    Author  : Rakeshvara Rao
    License : GNU GPL 3.0 or later

    This file outputs the trigram for given Telugu corpus.
"""
import sys
import pickle
from collections import defaultdict, Counter

from scripts.text2glyphs import process_line


if len(sys.argv) < 2:
    print("""Usage:
{0} out_file_prefix infile1 infile2 infile3 (etc.)
    Counts the unigram, bigram and trigram counts in the infiles and
    Writes them out to <out_file_prefix>.123.pkl
    Also shows the counts in respective .txt files.
    """.format(sys.argv[0]))
    sys.exit()

out_file_prefix = sys.argv[1]

# ############################################ Build the dictionaries

beg_line, end_line = ' ', ' '
gram1 = Counter()
gram2 = defaultdict(Counter)
gram3 = defaultdict(lambda: defaultdict(Counter))

for txt_file_name in sys.argv[2:]:
    corpus = open(txt_file_name)
    iline = 0

    for line in corpus:
        line = beg_line + line.rstrip() + end_line
        glyps = process_line(line)

        for i in range(len(glyps) - 2):
            a, b, c = glyps[i:i + 3]
            gram1[a] += 1
            gram2[a][b] += 1
            gram3[a][b][c] += 1

        # Add the last two glyphs
        if len(glyps) > 1:
            y, z = glyps[-2:]
            gram1[y] += 1
            gram1[z] += 1
            gram2[y][z] += 1
        elif len(glyps) > 0:
            gram1[glyps[-1]] += 1

        iline += 1
        if iline % 1000 == 0:
            print(txt_file_name, iline)

    corpus.close()

# ####################################### Remove special nature


def dictify(gram, levels):
    if levels == 3:  # Trigram
        for a in gram:
            for b in gram[a]:
                gram[a][b] = dict(gram[a][b])

    if levels >= 2:  # Bigram
        for a in gram:
            gram[a] = dict(gram[a])

    gram = dict(gram)
    return gram


gram1 = dictify(gram1, 1)
gram2 = dictify(gram2, 2)
gram3 = dictify(gram3, 3)

####################################### Normalize

def normalize(gram, levels):
    if levels == 1:  # Unigram
        total = sum(gram.values())
        for a in gram:
            gram[a] /= total

    elif levels == 2:  # Bigram
        for a in gram:
            total = sum(gram[a].values())
            for b in gram[a]:
                gram[a][b] /= total

    elif levels == 3:  # Trigram
        for a in gram:
            for b in gram[a]:
                total = sum(gram[a][b].values())
                for c in gram[a][b]:
                    gram[a][b][c] /= total


############################################## Dump Pickle
with open(out_file_prefix + '.123.pkl', 'wb') as f:
    pickle.dump([gram1, gram2, gram3], f)

if False:
    with open(out_file_prefix + '.uni.pkl', 'wb') as f:
        pickle.dump(dict(gram1), f)

    with open(out_file_prefix + '.bi.pkl', 'wb') as f:
        pickle.dump(dict(gram2), f)

    with open(out_file_prefix + '.tri.pkl', 'wb') as f:
        pickle.dump(dict(gram3), f)


############################################## Dump  txt


def sort(dic):
    return sorted(dic.items(), key=lambda x: x[0])

# Unigram
with open(out_file_prefix + '.uni.txt', 'w') as funi:
    for a, count in sort(gram1):
        funi.write('\n{} : {}'.format(a, count))

# Bigram
with open(out_file_prefix + '.bi.txt', 'w') as fbi:
    for a, d in sort(gram2):
        fbi.write('\n{} {}: {}'.format('*' * 40, a, len(d)))
        for b, count in sort(d):
            fbi.write('\n{} {} : {}'.format(a, b, count))

# Trigram
with open(out_file_prefix + '.tri.txt', 'w') as ftri:
    for a, dd in sort(gram3):
        ftri.write('\n\n{} {}: {}'.format('#' * 60, a, len(dd)))
        for b, d in sort(dd):
            ftri.write('\n{} {} {}: {}'.format('-' * 30, a, b, len(d)))
            for c, count in sort(d):
                ftri.write('\n{} {} {} : {}'.format(a, b, c, count))
