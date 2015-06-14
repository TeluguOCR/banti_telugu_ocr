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
import ast
import base64
import bz2
import io
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from PIL import Image

import numpy as np
import theano

from theanet.neuralnet import NeuralNet
from iast_unicodes import LabelToUnicodeConverter

# ############################################# Helpers


def read_json_bz2(path2data):
    print("Loading ", path2data)
    bz2_fp = bz2.BZ2File(path2data, 'r')
    data = np.array(json.loads(bz2_fp.read().decode('utf-8')))
    bz2_fp.close()
    return data


def share(data, dtype=theano.config.floatX):
    return theano.shared(np.asarray(data, dtype), borrow=True)


# ############################################# Arguments
print(' '.join(sys.argv))
if len(sys.argv) < 4:
    print('''Usage:
        {} dataprefix labels.lbl nnetfile.pkl...
        '''.format(sys.argv[0]))
    sys.exit()

x_data_file = sys.argv[1] + '.images.bz2'
y_data_file = sys.argv[1] + '.labels.bz2'
meta_file = sys.argv[1] + '.meta.bz2'
aux_data_file = sys.argv[1] + '.lines.bz2'

codes_file = sys.argv[2]
neural_net_files = sys.argv[3:]

# ############################################# Load Codes
with open(codes_file, 'r') as codes_fp:
    codes = ast.literal_eval(codes_fp.read())
chars = LabelToUnicodeConverter(codes).onecode

############################################## Load Data
print("Loading data files...")
x_data = read_json_bz2(x_data_file)
y_data = read_json_bz2(y_data_file)
meta_data = read_json_bz2(meta_file)
try:
    aux = share(read_json_bz2(aux_data_file))
except FileNotFoundError:
    pass

x_data_int = x_data.astype(np.uint8)
x = share(x_data)
y = share(y_data, 'int32')
n_samples = x_data.shape[0]
n_classes = y_data.max()

############################################## Test on NN


def test_on_network(neural_net_file):
    print('Loading network configuration: ', neural_net_file)
    with open(neural_net_file, 'rb') as prm_pkl_file:
        net_prms = pickle.load(prm_pkl_file)

    print('Initializing the network...')
    ntwk = NeuralNet(**net_prms)

    batch_sz = net_prms['training_params']['BATCH_SZ']
    n_batches = n_samples // batch_sz
    n_training_batches = int(n_batches *
                             net_prms['training_params']['TRAIN_ON_FRACTION'])

    print('Compiling Test Model...')
    tester = ntwk.get_test_model(x, y, aux, preds_feats=True)

    print('Classifying images...')
    counts = defaultdict(int)
    wrongs = defaultdict(int)
    errors = defaultdict(lambda: defaultdict(list))
    train_test_errs = [0, 0]

    for ibatch in range(n_batches):
        sym_err_rate, aux_stat, logprobs, predictions = tester(ibatch)

        for j in range(batch_sz):
            index = ibatch * batch_sz + j
            truth = y_data[index]
            logprob, guess = logprobs[j], predictions[j]
            counts[chars[truth]] += 1

            if guess != truth:
                train_test_errs[ibatch >= n_training_batches] += 1
                wrongs[chars[truth]] += 1
                rank_of_truth = sum(logprob > logprob[truth])
                prob_of_truth = int(100 * np.exp(logprob[truth]))
                prob_of_first = int(100 * np.exp(np.max(logprob)))
                errors[chars[truth]][chars[guess]].append(
                    (index, rank_of_truth, prob_of_truth, prob_of_first))

        if ibatch % 200 == 0 or ibatch == n_batches - 1:
            print('{} of {} batches. Errors Train:{} Test:{}'.format(
                ibatch + 1, n_batches, *train_test_errs))
            #if i == 100: break

    cum_err_rate = sum(wrongs.values()) / sum(counts.values())
    train_test_errs[0] /= n_training_batches * batch_sz
    train_test_errs[1] /= (n_batches - n_training_batches) * batch_sz
    print("Error Rates Cum:{:.2%} Train:{:.2%} Test:{:.2%}".format(
        cum_err_rate, *train_test_errs))

    ####################### Speed Check
    n_time_bathces = 100
    print("Checking speed: ", end="")
    start = time.time()
    for ibatch in range(n_time_bathces):
        tester(ibatch)
    elapsed = time.time() - start
    avg_time = 1000 * elapsed / (n_time_bathces * batch_sz)
    print("{:.3f}ms per glyph on {}".format(avg_time, theano.config.device))

    return {"ntwk": ntwk,
            "cum_err_rate": cum_err_rate,
            "train_test_errs": train_test_errs,
            "avg_time": avg_time,
            "wrongs": wrongs,
            "counts": counts,
            "errors": errors, }

############################################# HTML


def html(neural_net_file, ntwk,
         cum_err_rate, train_test_errs, avg_time,
         wrongs, counts, errors):
    head = '''<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body>
<h2>Banti Neural Network Errors</h2>
Dataset: <font face="monospace" color="blue">{0}</font></br>
Neural Net: <font face="monospace" color="blue">{1}</font></br></br>
<h4>Error Rates</h4>
<font face="monospace" color="brown">{2:.2%}</font> cumulative</br>
<font face="monospace" color="brown">{3[0]:.2%}</font> training</br>
<font face="monospace" color="brown">{3[1]:.2%}</font> test</br>
<h4>Speed</h4>
<font face="monospace" color="green">{4:.3f}</font>ms per glyph on
<font face="monospace" color="green">{5}</font>
<h4>Network</h4>
<h5>Specified Parameters:</h5><pre>{6}</pre>
<h5>Training Parameters:</h5><pre>{7}</pre>
<h5>Weights:</h5><pre>{8}</pre>
<h5>Generated Network:</h5><pre>{9}</pre>
<h4>Legend</h4><pre>sl#) [true_class]: ([wrongs] of [tested] = [error]%)</br>
[false] [image_shown]([rank], [true_probability]% vs [false_probability]%)...
</pre>
<h4>Results</h4>
'''.format(x_data_file, neural_net_file,
           cum_err_rate, train_test_errs,
           avg_time, theano.config.device,
           ntwk.get_layers_info(),
           ntwk.get_training_params_info(),
           ntwk.get_wts_info(detailed=True),
           ntwk)

    filler_main = '\n</br><p> {}) {} ({} of {} = {:.2f}%)</p>'
    filler_sub = '\n</br>{}'
    filler_img = '\n<img src="data:image/png;base64,{0}" title="{1} {2}"/> ' \
                 '({3}, {4} vs. {5} )'
    tail = '\n</body></html>'

    ####################### Write HTML
    out_file_name = x_data_file.replace('.bz2', '.with.') + \
                    os.path.basename(neural_net_file.replace('.pkl', '.html'))
    print('Compiling output html file ', out_file_name, end="\n\n")
    out_file = open(out_file_name, 'w', )
    out_file.write(head)

    for ibatch in range(n_classes):
        # Write Summary
        u = chars[ibatch]
        error_rate = 100.0 * wrongs[u] / counts[u] if counts[u] > 0 else 0.
        out_file.write(
            filler_main.format(ibatch, u, wrongs[u], counts[u], error_rate))

        # Write each bad classification
        for k, v in errors[u].items():
            out_file.write(filler_sub.format(k))

            for entry, rank_of_truth, prob_truth, prob_max in v:
                img = Image.fromarray(255 * (1 - x_data_int[entry]))
                buf = io.BytesIO()
                img.save(buf, format='BMP')
                im64 = base64.b64encode(buf.getvalue())

                out_file.write(filler_img.format(im64.decode("ascii"),
                                                 meta_data[entry][0],
                                                 meta_data[entry][1],
                                                 rank_of_truth,
                                                 prob_max,
                                                 prob_truth, ))

    out_file.write(tail)
    out_file.close()

###################################### Process all NNfiles
for nnfile in neural_net_files:
    try:
        ret = test_on_network(nnfile)
        html(nnfile, **ret)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print("Unexpected error:", sys.exc_info()[0])