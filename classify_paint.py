import ast
import os
import pickle
import sys
from bantry import BantryFile, Bantry, Space
from data.categorize import tile_raster_images
from PIL import Image as im
import numpy as np
from theanet.neuralnet import NeuralNet
from iast_unicodes import LabelToUnicodeConverter

############################################# Arguments
from scaler import ScalerFactory

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
chars = LabelToUnicodeConverter(labellings).onecode

############################################# Init Network
Bantry.scaler = ScalerFactory(scaler_prms)
bf = BantryFile(banti_file_name)

nnet_prms['training_params']['BATCH_SZ'] = 1
ntwk = NeuralNet(**nnet_prms)
tester = ntwk.get_data_test_model(go_nuts=True)

############################################# Image saver
dir_name = os.path.basename(nnet_prms_file_name)[:7] + '/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
namer = (dir_name + '{:03d}_{}_{:02d}.png').format
print("Look for me in :", dir_name)

def saver(outs, ch, debug=False):
    saver.index += 1
    for i, out in enumerate(outs):
        if out.ndim == 2:
            n_nodes = out.shape[1]
            w = n_nodes // int(np.sqrt(n_nodes))
            h = np.ceil(float(n_nodes) / w)
            buffer = np.full((1, w*h-n_nodes), .5)
            out = np.concatenate((out, buffer), 1).reshape((1, h, w))
        elif out.ndim == 4:
            out = out[0]

        if debug:
            print("{:7.3f} {:7.3f} {:7.3f}".format(out.max(), out.mean(), out.min()))

        im.fromarray(tile_raster_images(out, zm=2,
                                        make_white=True)
        ).save(namer(saver.index, chars[ch], i))

    if debug:
        print()

saver.index = 0

############################################# Read glyphs & classify
print("Classifying...")
for line_bantries in bf.file_bantries:
    for bantree in line_bantries:
        if bantree is Space:
            continue

        scaled_glp = bantree.scaled
        img = scaled_glp.pix.astype('float32').reshape((1,)+scaled_glp.pix.shape)

        if ntwk.takes_aux():
            dtopbot = scaled_glp.dtop, scaled_glp.dbot
            aux_data = np.array([[dtopbot, dtopbot]], dtype='float32')
            logprobs_or_feats, preds, *layer_outs = tester(img, aux_data)
        else:
            logprobs_or_feats, preds, *layer_outs = tester(img)

        saver(layer_outs, np.argmax(logprobs_or_feats))
    print("Saved {:4d} Images".format(saver.index))