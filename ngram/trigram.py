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

if len(sys.argv) < 2:
    print("""Usage:
        {0} input_text_file [show_trigram]
    Program counts the frequency of each Telugu letter following another and 
    writes out a binary <input_text_file>.bigram using pickle.
    show_trigram when supplied, a text file with the bigram is shown.
    """.format(sys.argv[0]))
    sys.exit()

beg_line, end_line = '$', '$'

# Build the dictionaries
tel_dump = open(sys.argv[1])

tridict = {}
iline = 0
for line in tel_dump:
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

tel_dump.close()

# Dump Pickle
import pickle
with open(sys.argv[1]+'.trigram', 'wb') as f:
    pickle.dump(dict(tridict), f)

# Dump  txt
if len(sys.argv) < 3:
    sys.exit()

fout = open('/tmp/trigram.txt', 'w')  
for k1, dd in sorted(tridict.items(), key=lambda x: x[0]):
    for k2, d in sorted(dd.items(), key=lambda x: x[0]):
        for k3, count in sorted(d.items(), key=lambda x: x[0]):
            fout.write('\n'+k1+":"+k2+":"+k3+":"+str(count))
        fout.write('\n.......')
    fout.write('\n\n............................................')
fout.close()

# Show
import os
os.system('gedit /tmp/trigram.txt &')
