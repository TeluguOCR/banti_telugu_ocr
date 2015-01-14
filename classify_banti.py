#!/usr/bin/python
from __future__ import print_function
import ast
import pickle
import sys

import numpy as np
from glyph.bantireader import BantiReader
from theanet.neuralnet import NeuralNet
from lookup import iast2uni

############################################# Arguments

if len(sys.argv) < 2:
    print("""Usage:
    {} neuralnet_params.pkl banti_output.box scaler_params.scl codes.lbl
    """.format(sys.argv[0]))
    sys.exit()

nnet_prms_file_name = sys.argv[1]
banti_file_name = sys.argv[2]
scaler_prms_file = sys.argv[3]
labelings_file_name = sys.argv[4]

############################################# Load Params
with open(scaler_prms_file, 'r') as sfp:
    scaler_prms = ast.literal_eval(sfp.read())

with open(nnet_prms_file_name, 'rb') as nnet_prms_file:
    nnet_prms = pickle.load(nnet_prms_file)

with open(labelings_file_name, 'r') as labels_fp:
    labellings = ast.literal_eval(labels_fp.read())

############################################# Init Network
gg = BantiReader(banti_file_name, scaler_prms)

ht = gg.scaler.params.TOTHT
nnet_prms['training_params']['BATCH_SZ'] = 1
nnet_prms['layers'][0][1]['img_sz'] = ht
ntwk = NeuralNet(nnet_prms['layers'],
                 nnet_prms['training_params'],
                 nnet_prms['allwts'])
tester = ntwk.get_data_test_model()

output = []

############################################# Read glyphs & classify
print("Classifying...")
for nsamples, metas, data in gg():
    for meta, scaled_glp in zip(metas, data):
        img = scaled_glp.pix.astype('float32').reshape((1, ht, ht))
        line, word, aux0, aux1 = meta

        if ntwk.takes_aux():
            aux_data = [[(aux0/ht, aux1/ht), (aux0/ht, aux1/ht)]]
            logprobs_or_feats, preds = tester(img, aux_data)
        else:
            logprobs_or_feats, preds = tester(img)

        output.append((line, word, preds, logprobs_or_feats))


############################################# Helpers
reverse_labels = dict((v, k) for k, v in labellings.items())


def index_to_char(index):
    try:
        return iast2uni[reverse_labels[index]]
    except KeyError:
        print('Failed to index to character ', index, reverse_labels[index])
        return '#'


def get_best_n(logprobab, n=5):
    topn = np.argsort(logprobab)[:-1 - n:-1]
    ret = '\n'
    for i in range(n):
        ret += '{} {:2.0f}\t'.format(index_to_char(topn[i]),
                                     100 * np.exp(logprobab[topn[i]]))
    return ret

############################################# Generate Text and Stats

best_match = ''
stats = '\n'
linenum = -1
wordnum = 0
for line, word, pred, logprob in output:
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

    best_match += index_to_char(pred[0])
    stats += get_best_n(logprob[0])

############################################# Write to text file
out_file_name = banti_file_name.replace('.box', '.txt')
print('Writing out put to ', out_file_name)
with open(out_file_name, 'w') as out_file:
    out_file.write(best_match + stats)