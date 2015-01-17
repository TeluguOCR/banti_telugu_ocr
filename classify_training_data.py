#!/usr/bin/python
"""
This script ...
    Takes: 
        parameters.pkl from a theanet training instance,
        Data files x, y, auxillary,
    Builds:
        A neural net with specified parameters,
    Runs:
        The training data images through the network,
    Reports:
        The errors made on the data samples,
    As:
        A neat html file with images and other metrics.
"""
from __future__ import print_function
import ast
import base64
import os
import pickle
import StringIO
import sys
from collections import defaultdict

import numpy as np
import theano
from PIL import Image

from theanet.neuralnet import NeuralNet
from iast_unicodes import get_index_to_char_converter

############################################## Helpers


def load_data(path2data):
    import bz2, json, contextlib
    with contextlib.closing(bz2.BZ2File(path2data, 'rb')) as f:
        return np.array(json.load(f))


def share(data, dtype=theano.config.floatX):
    return theano.shared(np.asarray(data, dtype), borrow=True)


############################################## Arguments
if len(sys.argv) < 4:
    print('''Usage:
        {} prms.pkl data.x.bz2 data.y.bz2 data.meta.bz2 labels.lbl [data.lines.bz2]
        '''.format(sys.argv[0]))
    sys.exit()

neural_net_file = sys.argv[1]
x_data_file = sys.argv[2]
y_data_file = sys.argv[3]
meta_file = sys.argv[4]
labels_file = sys.argv[5]
aux_data_file = sys.argv[6] if len(sys.argv) > 6 else None

############################################## Load NN
print('Loading network configuration...')
with open(neural_net_file, 'rb') as prm_pkl_file:
    net_prms = pickle.load(prm_pkl_file)

print('Initializing the network...')
ntwk = NeuralNet(**net_prms)

############################################## Load Codes
with open(labels_file, 'r') as labels_fp:
    labellings = ast.literal_eval(labels_fp.read())
index_to_char = get_index_to_char_converter(labellings)

############################################## Load Data
print("Loading data files...")
x_data = load_data(x_data_file)
y_data = load_data(y_data_file)
meta_data = load_data(meta_file)
x = share(x_data)
y = share(y_data, 'int32')
if aux_data_file:
    aux = share(load_data(aux_data_file))
else:
    aux = None

n_trin = x_data.shape[0]
batch_sz = net_prms['training_params']['BATCH_SZ']
n_trin_bth = n_trin / batch_sz
n_classes = y_data.max()

############################################## Compile Test Function
tester = ntwk.get_test_model(x, y, aux, preds_feats=True)


############################################## Classify
print('Classifying images...')
counts = defaultdict(int)
wrongs = defaultdict(int)
errors = defaultdict(lambda: defaultdict(list))

for i in range(n_trin_bth):
    sym_err_rate, aux_stat, logprobs, predictions = tester(i)

    for j in range(batch_sz):
        index = i * batch_sz + j
        truth = y_data[index]
        logprob, guess = logprobs[j], predictions[j]
        counts[index_to_char(truth)] += 1

        if guess != truth:
            wrongs[index_to_char(truth)] += 1
            rank_of_truth = sum(logprob > logprob[truth])
            prob_of_truth = int(100 * np.exp(logprob[truth]))
            prob_of_first = int(100 * np.exp(np.max(logprob)))
            errors[index_to_char(truth)][index_to_char(guess)].append(
                (index, rank_of_truth, prob_of_truth, prob_of_first))

    if i % 20 == 0:
        print('{} of {} batches done'.format(i + 1, n_trin_bth))
        #if i == 100: break

####################### HTML strings

head = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head> 
<body><h1>Errors for file {} using the parameters {}</h1>
</br>#) &lt;glyph>:(&lt;wrongs> of &lt;tested> = &lt;error%>)
</br>&lt;recognized_as>&lt;culprit_image>(rank, truth's probability% vs max's
probability%) ...repeat...
'''.format(x_data_file, neural_net_file)
filler_main = '\n</br><p> {}) {} ({} of {} = {:.2f}%)</p>'
filler_sub = '\n</br>{}'
filler_img = '\n<img src="data:image/png;base64,{}" title=""/> ' \
             '({} {} - {}, {} vs. {} )'
tail = '\n</body></html>'


####################### Write HTML
x_data = x_data.astype(np.uint8)

out_file_name = x_data_file.replace('.bz2', '.with.') + \
                os.path.basename(neural_net_file.replace('.pkl', '.html'))
print('Compiling output html file ', out_file_name)
out_file = open(out_file_name, 'w',)
out_file.write(head)

for i in range(n_classes):
    # Write Summary
    u = index_to_char(i)
    error_rate = 100.0 * wrongs[u] / counts[u] if counts[u] > 0 else 0.
    out_file.write(filler_main.format(i, u, wrongs[u], counts[u], error_rate))

    # Write each bad classification
    for k, v in errors[u].items():
        out_file.write(filler_sub.format(k))

        for entry, rank_of_truth, prob_truth, prob_max in v:
            img = Image.fromarray(255 * (1-x_data[entry]))
            buf = StringIO.StringIO()
            img.save(buf, format='BMP')
            im64 = base64.b64encode(buf.getvalue())

            out_file.write(filler_img.format(im64,
                                             meta_data[entry][0],
                                             meta_data[entry][1],
                                             rank_of_truth,
                                             prob_max,
                                             prob_truth,))

out_file.write(tail)
out_file.close()
