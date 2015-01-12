#!/usr/bin/python
from __future__ import print_function
import ast
import codecs
import pickle
import sys

import numpy as np
from glyph.bantireader import BantiReader
from theanet.neuralnet import NeuralNet

############################################# Arguments

if len(sys.argv) < 2:
    print("""Usage:
    {} neuralnet_params.pkl banti_output.box scaler_params.scl
    """.format(sys.argv[0]))
    sys.exit()

nnet_prms_file_name = sys.argv[1]
banti_file_name = sys.argv[2]
scaler_prms_file = sys.argv[3]


############################################# Load Params
with open(scaler_prms_file, 'r') as sfp:
    scaler_prms = ast.literal_eval(sfp.read())

with open(nnet_prms_file_name, 'rb') as nnet_prms_file:
    nnet_prms = pickle.load(nnet_prms_file)

ht = scaler_prms['TOTHT']
nnet_prms['training_params']['BATCH_SZ'] = 1
nnet_prms['layers'][0][1]['img_sz'] = ht
ntwk = NeuralNet(**nnet_prms)
tester = ntwk.get_data_test_model()
output = []

gg = BantiReader(banti_file_name, scaler_prms)
for nsamples, metas, data in gg():
    for meta, scaled_glp in zip(metas, data):
        img = scaled_glp.pix
        line, word, aux0, aux1 = meta

        if ntwk.takes_aux():
            aux_data = [(aux0/ht, aux1/ht), (aux0/ht, aux1/ht)]
            logprobs_or_feats, preds = tester(img, aux_data)
        else:
            logprobs_or_feats, preds = tester(img)

        output.append((line, word, preds, logprobs_or_feats))

def get_best_n(logprob, n=5):
    topn = np.argsort(logprob)[:-1 - n:-1]
    ret = '\n'
    for i in range(n):
        ret += unicodes[topn[i]] + ' {:2.0f}\t'.format(
            100 * np.exp(logprob[topn[i]]))
    return ret


best_match = ''
stats = '\n'
linenum = -1
wordnum = 0
for meta, pred, logprob in output:
    if meta[0] > linenum:
        best_match += '\n'
        stats += '\n'
        linenum += 1
        wordnum = 0
    if meta[1] > wordnum:
        best_match += ' '
        wordnum += 1

    best_match += unicodes[pred]
    stats += get_best_n(logprob)

with codecs.open(banti_file_name[:-4] + '.txt', 'w', 'utf-8') as out_file:
    out_file.write(best_match + stats)
