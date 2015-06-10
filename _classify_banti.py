#!/usr/bin/python
from __future__ import print_function
import ast
import pickle
import sys
import logging

import numpy as np
from theanet.neuralnet import NeuralNet

from glyph.bantireader import BantiReader
from iast_unicodes import get_index_to_char_converter

import ngram._path as path
from ngram._bantry import Bantry, process_line_bantires
from ngram.post_process import post_process

############################################# Arguments
default_ngram = "ngram/mega.123.pkl"

if len(sys.argv) < 5:
    print("Usage:"
    "\n{0} neuralnet_params.pkl banti_output.box scaler_params.scl codes.lbl "
    "[gram.tri.pkl='{1}']"
    "\n\te.g:- {0} trained.pkl sample_images/praasa.box "
    "scalings/relative48.scl labelings/alphacodes.lbl"
    "".format(sys.argv[0], default_ngram))
    sys.exit()

nnet_prms_file_name = sys.argv[1]
banti_file_name = sys.argv[2]
scaler_prms_file = sys.argv[3]
labelings_file_name = sys.argv[4]
if len(sys.argv) > 5 and sys.argv[5].endswith(".pkl"):
    trigram_file = sys.argv[5]
else:
    trigram_file = default_ngram

logging.basicConfig(filename=banti_file_name.replace('.box', '.log'),
                    level=logging.INFO,
                    filemode="w")
if sys.argv[-1] != "1":
    print("Logging disabled.")
    logging.disable(logging.CRITICAL)
else:
    print("logging enabled")
logging.info(' '.join(sys.argv))

def get_best_n(logprobab, n=5):
    topn = np.argsort(logprobab)[:-1 - n:-1]
    ret = ''
    for i in range(n):
        ret += '{} {:2.0f}\t'.format(index_to_char(topn[i]),
                                     100 * np.exp(logprobab[topn[i]]))
    return ret

############################################# Generate Text and Stats

best_match = ''
stats = '\n'
linenum = -1
wordnum = 0
for line, word, pred, logprob, aux0, aux1 in output:
    if line > linenum:
        best_match += '\n'
        stats += '\n'
        linenum += 1
        wordnum = 0
        print('\n', linenum, ':', end=' ')
    if word > wordnum:
        best_match += ' '
        wordnum += 1
        print(wordnum, end=' ')

    best_match += index_to_char(pred)
    stats += '\n@({:3d}, {:3d}) '.format(aux0, aux1) + get_best_n(logprob)

print()

############################################# Write to text file
out_file_name = banti_file_name.replace('.box', '.txt')
print('Writing output to ', out_file_name)
with open(out_file_name, 'w', encoding='utf-8') as out_file:
    out_file.write(post_process(best_match))

out_file_name = banti_file_name.replace('.box', '.matches')
print('Writing matches to ', out_file_name)
with open(out_file_name, 'w', encoding='utf-8') as out_file:
    out_file.write(stats)


####################################################### Try N-gram
nclasses = output[-1][-3].size
codes = np.array([index_to_char(i) for i in range(nclasses)])

path.priorer.set_trigram(trigram_file)

curr_line = 0
line_bantries = []
decency = 5

out_file_name = banti_file_name.replace('.box', '.gram.txt')
print('Writing ngrammed output to ', out_file_name)
ngramout = open(out_file_name, 'w', encoding='utf-8')


for line, word, preds, logprobs, _, _ in output:
    if line < curr_line:
        raise (ValueError, "Line number can not go down {}->{}".format(
            curr_line, line))

    decent = np.argpartition(logprobs, -decency)[-decency:]
    e = Bantry(line, word, zip(codes[decent], logprobs[decent]))

    if line == curr_line:
        line_bantries.append(e)

    else:
        processed = process_line_bantires(line_bantries)
        ngramout.write(post_process(processed))

        line_bantries = [e]
        curr_line = line

processed = process_line_bantires(line_bantries)
ngramout.write(post_process(processed))

ngramout.close()