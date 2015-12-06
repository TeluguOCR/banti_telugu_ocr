#! /usr/bin/python
# -*- coding: utf-8 -*-
import ast
import pickle
import numpy as np
import os
import socket
import sys
import bz2
import json
from datetime import datetime

import theano as th
import theanet.neuralnet as nn

################################ HELPER FUNCTIONS ############################


def read_json_bz2(path2data):
    bz2_fp = bz2.BZ2File(path2data, 'r')
    data = np.array(json.loads(bz2_fp.read().decode('utf-8')))
    bz2_fp.close()
    return data


def share(data, dtype=th.config.floatX, borrow=True):
    return th.shared(np.asarray(data, dtype), borrow=borrow)


class WrapOut:
    def __init__(self, use_file, name=''):
        self.name = name
        self.use_file = use_file
        if use_file:
            self.stream = open(name, 'w', 1)
        else:
            self.stream = sys.stdout

    def write(self, data):
        self.stream.write(data)

    def forceflush(self):
        if self.use_file:
            self.stream.close()
            self.stream = open(self.name, 'a', 1)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

################################### MAIN CODE ################################

if len(sys.argv) < 4:
    print('Usage:', sys.argv[0],
          ''' <x.bz2> <y.bz2> <params_file(s)> [auxillary.bz2] [redirect=0]
    .bz2 files contain the samples and the output classes as generated
        by the gen_cnn_data.py script (or the like).
    params_file(s) :
        Parameters for the NeuralNet
        - params_file.py  : contains the initialization code
        - params_file.pkl : pickled file from a previous run (has wts too).
    redirect:
        1 - redirect stdout to a <SEED>.txt file
    ''')
    sys.exit()

imgs_file_name = sys.argv[1]
lbls_file_name = sys.argv[2]
prms_file_name = sys.argv[3]
if len(sys.argv) > 4 and sys.argv[4].endswith('.bz2'):
    aux_file_name = sys.argv[4]
else:
    aux_file_name = None

##########################################  Import Parameters

if prms_file_name.endswith('.pkl'):
    with open(prms_file_name, 'rb') as f:
        params = pickle.load(f)
else:
    with open(prms_file_name, 'r') as f:
        params = ast.literal_eval(f.read())

layers = params['layers']
tr_prms = params['training_params']
try:
    allwts = params['allwts']
except KeyError:
    allwts = None

## Init SEED
if (not 'SEED' in tr_prms) or (tr_prms['SEED'] is None):
    tr_prms['SEED'] = np.random.randint(0, 1e6)

out_file_head = os.path.basename(prms_file_name,).replace(
    os.path.splitext(prms_file_name)[1], "_{:06d}".format(tr_prms['SEED']))

if sys.argv[-1] is '1':
    print("Printing output to {}.txt".format(out_file_head), file=sys.stderr)
    sys.stdout = WrapOut(True, out_file_head + '.txt')
else:
    sys.stdout = WrapOut(False)


##########################################  Print Parameters

print(' '.join(sys.argv), file=sys.stderr)
print(' '.join(sys.argv))
print('Time   :' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('Device : {} ({})'.format(th.config.device, th.config.floatX))
print('Host   :', socket.gethostname())

print(nn.get_layers_info(layers))
print(nn.get_training_params_info(tr_prms))

##########################################  Load Data

print("\nLoading the data ...")
sys.stdout.forceflush()
data_x = read_json_bz2(imgs_file_name)
data_y = read_json_bz2(lbls_file_name)
if data_x.ndim == 3:
    data_x = np.expand_dims(data_x, axis=1)

print("X (samples, dimensions): {} {}KB\n"
      "X (min, max) : {} {}\n"
      "Y (samples, dimensions): {} {}KB\n"
      "Y (min, max) : {} {}".format(data_x.shape, data_x.nbytes // 1000,
                                    data_x.min(), data_x.max(),
                                    data_y.shape, data_y.nbytes // 1000,
                                    data_y.min(), data_y.max()))

batch_sz = tr_prms['BATCH_SZ']
corpus_sz, _, layers[0][1]['img_sz'], _ = data_x.shape

n_train = int(corpus_sz * tr_prms['TRAIN_ON_FRACTION'])

trin_x = share(data_x[:n_train, ])
test_x = share(data_x[n_train:, ])
trin_y = share(data_y[:n_train, ], 'int32')
test_y = share(data_y[n_train:, ], 'int32')

if aux_file_name:
    data_aux = read_json_bz2(aux_file_name)
    trin_aux = share(data_aux[:n_train, ])
    test_aux = share(data_aux[n_train:, ])
else:
    trin_aux, test_aux = None, None

print("\nInitializing the net ... ")
net = nn.NeuralNet(layers, tr_prms, allwts)
print(net)
print(net.get_wts_info(detailed=True).replace("\n\t", ""))

print("\nCompiling ... ")
training_fn = net.get_trin_model(trin_x, trin_y, trin_aux)
test_fn_tr = net.get_test_model(trin_x, trin_y, trin_aux)
test_fn_te = net.get_test_model(test_x, test_y, test_aux)

tr_corpus_sz = n_train
te_corpus_sz = corpus_sz - n_train
nEpochs = tr_prms['NUM_EPOCHS']
nTrBatches = tr_corpus_sz // batch_sz
nTeBatches = te_corpus_sz // batch_sz

############################################## MORE HELPERS 


def test_wrapper(nylist):
    sym_err, bit_err, n = 0., 0., 0
    for symdiff, bitdiff in nylist:
        sym_err += symdiff
        bit_err += bitdiff
        n += 1
    return 100 * sym_err / n, 100 * bit_err / n


if net.tr_layers[-1].kind == 'LOGIT':
    aux_err_name = 'BitErr'
else:
    aux_err_name = 'P(MLE)'


def get_test_indices(tot_samps, bth_samps=tr_prms['TEST_SAMP_SZ']):
    n_bths_each = int(bth_samps / batch_sz)
    n_bths_all = int(tot_samps / batch_sz)
    cur = 0
    while True:
        yield [i % n_bths_all for i in range(cur, cur + n_bths_each)]
        cur = (cur + n_bths_each) % n_bths_all


test_indices = get_test_indices(te_corpus_sz)
trin_indices = get_test_indices(tr_corpus_sz)
pickle_file_name = out_file_head + '_{:02.0f}.pkl'
saved_file_name = None


def do_test():
    global saved_file_name
    test_err, aux_test_err = test_wrapper(test_fn_te(i)
                                          for i in next(test_indices))
    trin_err, aux_trin_err = test_wrapper(test_fn_tr(i)
                                          for i in next(trin_indices))
    print("{:5.2f}%  ({:5.2f}%)      {:5.2f}%  ({:5.2f}%)".format(
        trin_err, aux_trin_err, test_err, aux_test_err))
    sys.stdout.forceflush()

    if saved_file_name:
        os.remove(saved_file_name)

    saved_file_name = pickle_file_name.format(test_err)
    with open(saved_file_name, 'wb') as pkl_file:
        pickle.dump(net.get_init_params(), pkl_file, -1)

############################################ Training Loop

print("Training ...")
print("Epoch   Cost  Tr_Error Tr_{0}    Te_Error Te_{0}".format(aux_err_name))
for epoch in range(nEpochs):
    total_cost = 0

    for ibatch in range(nTrBatches):
        output = training_fn(ibatch)
        total_cost += output[0]

        if np.isnan(total_cost):
            print("Epoch:{} Iteration:{}".format(epoch, ibatch))
            print(net.get_wts_info(detailed=True))
            raise ZeroDivisionError("Nan cost at Epoch:{} Iteration:{}"
                                    "".format(epoch, ibatch))

    if epoch % tr_prms['EPOCHS_TO_TEST'] == 0:
        print("{:3d} {:>8.2f}".format(net.get_epoch(), total_cost), end='    ')
        do_test()
        if  total_cost > 1e6:
            print(net.get_wts_info(detailed=True))

    net.inc_epoch_set_rate()

########################################## Final Error Rates

test_err, aux_test_err = test_wrapper(test_fn_te(i)
                                      for i in range(te_corpus_sz//batch_sz))
trin_err, aux_trin_err = test_wrapper(test_fn_tr(i)
                                      for i in range(tr_corpus_sz//batch_sz))

print("{:3d} {:>8.2f}".format(net.get_epoch(), 0), end='    ')
print("{:5.2f}%  ({:5.2f}%)      {:5.2f}%  ({:5.2f}%)".format(
        trin_err, aux_trin_err, test_err, aux_test_err))
