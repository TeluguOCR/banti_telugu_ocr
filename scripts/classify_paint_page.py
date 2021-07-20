"""
This script process Telugu text and paints all the glyphs as they go through the Neural Network
    Takes:
        parameters.pkl from a theanet training instance,
        Input Image (as tif, box, etc.),
        Scaling parameters
        Labeles of classes
    Builds:
        A neural net with specified parameters,
    Processes:
        The page with banti,
    Runs:
        The glyphs through the network,
    Renders:
        The images of the glyph as they go through the network,
    As:
        A neat html file with images and other metrics.
"""
import ast
import os
import pickle
import sys

from PIL import Image as im
import numpy as np
from theanet.neuralnet import NeuralNet

from banti.iast_unicodes import LabelToUnicodeConverter
from banti.scaler import ScalerFactory
from banti.proglyph import ProGlyph, Space
from banti.processedpage import ProcessedPage
from scripts.tile import tile_raster_images, tile_zagged_columns

############################################# Arguments

if len(sys.argv) != 5:
    print("Usage:"
    "\n{0} neuralnet_params.pkl inpt.box/tiff scaler_params.scl codes.lbl "
    "\n\te.g:- {0} 0default.pkl sample_images/praasa.box scalings/relative48.scl labellings/alphacodes.lbl"
    "".format(sys.argv[0]))
    sys.exit()

_, nnet_prms_file_name, input_file_name, scaler_prms_file, labelings_file_name = sys.argv

############################################# Load Params

with open(scaler_prms_file, 'r') as sfp:
    scaler_prms = ast.literal_eval(sfp.read())

with open(nnet_prms_file_name, 'rb') as nnet_prms_file:
    nnet_prms = pickle.load(nnet_prms_file)

with open(labelings_file_name, encoding='utf-8') as labels_fp:
    labellings = ast.literal_eval(labels_fp.read())

chars = LabelToUnicodeConverter(labellings).onecode

############################################# Init Network
ProGlyph.scaler = ScalerFactory(scaler_prms)
procd_page = ProcessedPage(input_file_name)

nnet_prms['training_params']['BATCH_SZ'] = 1
ntwk = NeuralNet(**nnet_prms)
print(ntwk)
tester = ntwk.get_data_test_model(get_output_of_layers=range(ntwk.num_layers))

############################################# Image saver
dir_name = os.path.basename(nnet_prms_file_name)[:-7] + '/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
namer = (dir_name + '{:03d}_{}.png').format
print("Look for me in :", dir_name)


def saver(outs, ch, debug=True):
    saver.index += 1
    images = []
    for i, out in enumerate(outs):
        if out.ndim == 2:
            # Make one dimensional output of hidden layers into a matrix
            batch_sz, n_nodes = out.shape
            assert batch_sz == 1
            w = n_nodes // int(np.sqrt(n_nodes))
            h = int(np.ceil(float(n_nodes) / w))
            extra = np.full((1, w*h-n_nodes), 0)
            out = np.concatenate((out, extra), 1).reshape((1, h, w))
        elif out.ndim == 4:
            out = out[0]

        if debug:
            print(f"Max:{out.max():6.2f} Mean:{out.mean():6.2f} Min:{out.min():6.2f} Shape:{out.shape}")

        images.append(tile_raster_images(out, zm=2, make_white=True, global_normalize=True))

    im.fromarray(tile_zagged_columns(images, 2)).save(namer(saver.index, chars[ch]), compress_level=1)

    if debug:
        print()


saver.index = 0

############################################# Read glyphs & classify
print("Classifying...")
for line_pglyphs in procd_page.file_glyphs:
    for pglyph in line_pglyphs:
        if pglyph is Space:
            continue

        scaled_glp = pglyph.scaled
        img = scaled_glp.pix.astype('float32').reshape((1, 1,)+scaled_glp.pix.shape)

        if ntwk.takes_aux():
            dtopbot = scaled_glp.dtop, scaled_glp.dbot
            aux_data = np.array([[dtopbot, dtopbot]], dtype='float32')
            logprobs_or_feats, preds, *layer_outs = tester(img, aux_data)
        else:
            logprobs_or_feats, preds, *layer_outs = tester(img)

        saver(layer_outs, np.argmax(logprobs_or_feats), saver.index == 0)
    print(f"Saved images of {saver.index} glyphs")

print("Look for me in :", dir_name)
