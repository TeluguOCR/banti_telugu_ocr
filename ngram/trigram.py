#!/usr/bin/env python3
"""
    Project : Telugu OCR
    Author  : Rakeshvara Rao
    License : GNU GPL 3.0 or later

    This file outputs the trigram for given Telugu corpus.
    This uses module BantiParser
"""
import sys
from unicode2labels import process_line
import pickle

if len(sys.argv) < 2:
    print("""Usage:
        {0} input_text_file
    Program counts the frequency of each Telugu letter following another and 
    writes out a binary <input_text_file>.tri.pkl using pickle.
    """.format(sys.argv[0]))
    sys.exit()

txt_file_name = sys.argv[1]
txt_file_head = txt_file_name[:-4] if txt_file_name.endswith('.txt') \
    else txt_file_name

############################################# Build the dictionaries
corpus = open(txt_file_name)

beg_line, end_line = ' ', ' '
tridict = {}
iline = 0
for line in corpus:
    line = beg_line + line.rstrip() + end_line
    glyps = process_line(line)

    for i in range(len(glyps)-2):
        a = glyps[i]
        b = glyps[i+1]
        c = glyps[i+2]
        if not a in tridict:
            tridict[a] = {}

        if not b in tridict[a]:
            tridict[a][b] = {}

        if not c in tridict[a][b]:
            tridict[a][b][c] = 0

        tridict[a][b][c] += 1

    iline += 1
    if iline%1000 == 0:
        print(iline)

corpus.close()

############################################## Normalize
for a in tridict:
    for b in tridict[a]:
        total = sum(tridict[a][b].values())
        for c in tridict[a][b]:
            tridict[a][b][c] /= total


############################################## Dump Pickle
with open(txt_file_head+'.tri.pkl', 'wb') as f:
    pickle.dump(dict(tridict), f)


############################################## Dump  txt
def sort(dic):
    return sorted(dic.items(),  key=lambda x: x[0])

fout = open(txt_file_head+'.tri.txt', 'w')

for a, dd in sort(tridict):
    fout.write('\n\n{} {}: {}'.format('#'*60, a, len(dd)))

    for b, d in sort(dd):
        fout.write('\n{} {} {}: {}'.format('-'*30, a, b, len(d)))

        for c, count in sort(d):
            fout.write('\n{} {} {} : {:8.6f}'.format(a, b, c, count))

fout.close()
