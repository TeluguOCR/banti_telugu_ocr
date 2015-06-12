import ast
import os
import pickle
import sys
from math import ceil
from data.categorize import tile_raster_images
from PIL import Image as im
import numpy as np
from glyph.bantireader import BantiReader
from theanet.neuralnet import NeuralNet
from iast_unicodes import LabelToUnicodeConverter

############################################# Arguments

if len(sys.argv) < 5:
    print("Usage:"
    "\n{0} neuralnet_params.pkl banti_output.box scaler_params.scl codes.lbl "
    "\n\te.g:- {0} cnn_softaux_gold.pkl sample_images/praasa.box "
    "glyph/scalings/relative48.scl glyph/labellings/alphacodes.lbl"
    "".format(sys.argv[0]))
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

with open(labelings_file_name, encoding='utf-8') as labels_fp:
    labellings = ast.literal_eval(labels_fp.read())

# print(labellings)
index_to_char = LabelToUnicodeConverter(labellings)

############################################# Init Network
gg = BantiReader(banti_file_name, scaler_prms)

ht = gg.scaler.params.TOTHT
nnet_prms['training_params']['BATCH_SZ'] = 1
nnet_prms['layers'][0][1]['img_sz'] = ht
ntwk = NeuralNet(nnet_prms['layers'],
                 nnet_prms['training_params'],
                 nnet_prms['allwts'])
tester = ntwk.get_data_test_model(go_nuts=True)

output = []

############################################# Image saver
dir_name = os.path.basename(nnet_prms_file_name)[:7] + '/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
namer = (dir_name + '{:03d}_{}_{:02d}.png').format
print("Look for me in :", dir_name)

def saver(outs, ch):
    saver.index += 1
    for i, out in enumerate(outs):
        if out.ndim == 2:
            out = out.reshape((out.shape[1], 1, 1))
        elif out.ndim == 4:
            out = out[0]

        im.fromarray(tile_raster_images(out)).save(
            namer(saver.index, index_to_char(ch), i))

saver.index = 0

############################################# Read glyphs & classify
print("Classifying...")
for nsamples, metas, data in gg():
    for meta, scaled_glp in zip(metas, data):
        img = scaled_glp.pix.astype('float32').reshape((1, ht, ht))
        line, word, aux0, aux1 = meta

        if ntwk.takes_aux():
            aux_data = np.array([[(aux0/ht, aux1/ht), (aux0/ht, aux1/ht)]], dtype='float32')
            logprobs_or_feats, preds, *layer_outs = tester(img, aux_data)
        else:
            logprobs_or_feats, preds, *layer_outs = tester(img)

        output.append((line, word, preds[0], logprobs_or_feats[0], aux0, aux1))
        saver(layer_outs, np.argmax(logprobs_or_feats))
    print("Saved {:4d} Images".format(saver.index))

############################################# Helpers

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
out_file_name = banti_file_name.replace('.box', '.matches')
print('Writing matches to ', out_file_name)
with open(out_file_name, 'w', encoding='utf-8') as out_file:
    out_file.write(stats)

