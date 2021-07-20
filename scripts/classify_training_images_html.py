#! /usr/bin/python
# -*- coding: utf-8 -*-
import ast
import glob
import os
import pickle
import re
import sys
import io
import base64
import tarfile
from random import choice
from collections import defaultdict
from traceback import print_exc

import numpy as np
import theano
from PIL import Image

from banti.basicglyph import BasicGlyph
from banti.scaler import ScalerFactory
from theanet.neuralnet import NeuralNet
from banti.iast_unicodes import LabelToUnicodeConverter

################################### Process Files & Dirs #####################


def split_file_name(file_path):
    # Font_Style_ID_T_B_T_B* (Akshar_IT_4004018_-5_-28_-4_-27_-3_-26_-6_-29)
    file_name = os.path.basename(file_path)
    m = re.match('(.+?)_(..)_(.+?)(_.+_.+).tif', file_name)
    try:
        font = m.group(1)
        style = m.group(2)
        id_ = m.group(3)
        dtbs = list(map(int, m.group(4).split('_')[1:]))
        dtbpairs = [(dtbs[i], dtbs[i+1]) for i in range(0, len(dtbs), 2)]
        return font, style, id_, dtbpairs

    except ValueError:
        print("Bad filename : ", file_path)
        return '', '', '', []


class GlyphDir(object):
    def __init__(self, dir_path, default_scaler, default_sampler):
        glyphs = defaultdict(list)

        for file_path in os.listdir(dir_path):
            if not file_path.endswith('.tif'):
                print('Skipping non tiff file', file_path)
                continue

            full_path = os.path.join(dir_path, file_path)
            img = Image.open(full_path)
            font, style, id_, dtbpairs = split_file_name(file_path)
            glyph = BasicGlyph((img, dtbpairs))
            glyph.in_path = full_path
            glyphs[(font, style)].append(glyph)

        self.glyphs = glyphs
        self.dir = dir_path
        self.name = os.path.basename(dir_path)
        self.scaler = default_scaler
        self.default_getter = default_sampler

    def _scale(self, glyph, dtop, dbot):
        try:
            scaled_glyph = self.scaler(BasicGlyph((glyph.img, dtop, dbot)))
            scaled_glyph.in_path = glyph.in_path
        except ValueError:
            print("Bad hombre : ", glyph.in_path)
            return None
        return scaled_glyph

    def get_all(self, ):
        for font_style, glyph_list in list(self.glyphs.items()):
            for glyph in glyph_list:
                for dtop, dbot in glyph.dtopbot_pairs:
                    scaled_glyph = self._scale(glyph, dtop, dbot)
                    if scaled_glyph is not None:
                        yield scaled_glyph

    def get_one_per_file(self, ):
        for font_style, glyph_list in list(self.glyphs.items()):
            for glyph in glyph_list:
                dtop, dbot = choice(glyph.dtopbot_pairs)
                scaled_glyph = self._scale(glyph, dtop, dbot)
                if scaled_glyph is not None:
                    yield scaled_glyph

    def get_one_per_style(self, ):
        for font_style, glyph_list in list(self.glyphs.items()):
            glyph = choice(glyph_list)
            dtop, dbot = choice(glyph.dtopbot_pairs)
            scaled_glyph = self._scale(glyph, dtop, dbot)
            if scaled_glyph is not None:
                yield scaled_glyph

    def getter(self):
        return getattr(self, self.default_getter)()

################################# ZIP UNZIP #################################


def extract_tar(tar_name, target_dir):
    print("Extracting the tarfile", tar_name, "to", target_dir)
    data_tar = tarfile.open(tar_name)
    data_tar.extractall(path=target_dir)
    data_tar.close()


################################# MAIN #################################

if len(sys.argv) < 6:
    print('''Usage:
{0} <images.tar.gz/dir> <scaler.scl> <codes.lbl> <image_sampler> <neural_net>

    images.tar.gz/dir : contains the samples images in their directories,
        can be a tar file or an extracted directory.
    scaler.scl : The scaler paramerters dict in an ast readable file
    codes.lbl: map from glyph names to integers
    image_sampler: All, One per file, One per style, relative etc.

e.g.s:-
{0} images.tar.gz absolute64.scl alphacodes.lbl get_one_per_style abs68
{0} /tmp/tel_ocr/numbers absolute09.scl numbers.lbl get_all num32
{0} images.tar.gz relative48.scl alphacodes.lbl get_relative rel48
{0} /tmp/tel_ocr absolute64.scl mallicodes.lbl get_one_per_file abs68malli
    '''.format(sys.argv[0]))
    sys.exit(-1)

######################################## Process Arguments
_, tar_file, scaler_prms_file, codes_file, image_sampler = sys.argv[:5]
neural_net_files = sys.argv[5:]
dirs_dir = '/tmp/rakesha/'

######################################## Open parameter files
with open(scaler_prms_file) as images_fp:
    scaler_params = ast.literal_eval(images_fp.read())
    scaler = ScalerFactory(scaler_params)

lookup = LabelToUnicodeConverter(codes_file)
iast2idx = lookup.labels
idx2unic = lookup.onecode

################################### Extract tar and change to relevant sub-dir
if os.path.isdir(tar_file):
    dirs_dir = tar_file
else:
    extract_tar(tar_file, dirs_dir)

if len(os.listdir(dirs_dir)) == 1:
    dirs_dir = os.path.join(dirs_dir, os.listdir(dirs_dir)[0])
    assert os.path.isdir(dirs_dir)

################################### ################################### ###################################


def process_root_dir(root_dir):
    idir = 0

    for dir_name in sorted(os.listdir(root_dir)):
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path):
            print('Skipping non-directory ', dir_path)
            continue

        idir += 1
        print(idir, 'Processing Dir :', dir_path)

        glp_dir = GlyphDir(dir_path, scaler, image_sampler)
        label = iast2idx[glp_dir.name]

        for glyph in glp_dir.getter():
            if "Suguna" in glyph.in_path:
                continue
            glyph.label = label
            yield glyph


def classify(ntwk, tester, glyph):
    img = glyph.pix
    img = img.astype(theano.config.floatX).reshape((1, 1)+img.shape)

    if ntwk.takes_aux():
        dtopbot = glyph.dtop, glyph.dbot
        aux_data = np.array([[dtopbot, dtopbot]], dtype='float32')
        logprobs, preds = tester(img, aux_data)
    else:
        logprobs, preds = tester(img)
    return logprobs[0], preds[0]


def test_on_network(neural_net_file):
    print('Loading network configuration: ', neural_net_file)
    with open(neural_net_file, 'rb') as prm_pkl_file:
        net_prms = pickle.load(prm_pkl_file)

    print('Initializing the network...')
    net_prms['training_params']['BATCH_SZ'] = 1
    ntwk = NeuralNet(**net_prms)

    print('Compiling Test Model...')
    tester = ntwk.get_data_test_model()

    print('Classifying images...')
    counts = defaultdict(int)
    wrongs = defaultdict(int)
    errors = defaultdict(lambda: defaultdict(list))

    for glyph in process_root_dir(dirs_dir):
        logprob, guessi = classify(ntwk, tester, glyph)
        truei = glyph.label
        truec = idx2unic[truei]
        guessc = idx2unic[guessi]

        counts[truec] += 1
        if guessi != truei:
            wrongs[truec] += 1
            glyph.rank_true = sum(logprob > logprob[truei])
            glyph.ptruth = int(100 * np.exp(logprob[truei]))
            glyph.pguess = int(100 * np.exp(np.max(logprob)))
            glyph.guess_label = guessi
            errors[truec][guessc].append(glyph)

    cum_err_rate = sum(wrongs.values()) / sum(counts.values())
    print(f"Error Rates Cum:{cum_err_rate:.2%}")

    return {"neural_net_file": neural_net_file,
            "ntwk": ntwk,
            "cum_err_rate": cum_err_rate,
            "wrongs": wrongs,
            "counts": counts,
            "errors": errors, }

############################################# HTML


def html(neural_net_file, ntwk, cum_err_rate, wrongs, counts, errors):
    head = f'''<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body>
<h2>Neural Network Errors</h2>
Dataset: <font face="monospace" color="blue">{dirs_dir}</font></br>
Neural Net: <font face="monospace" color="blue">{neural_net_file}</font></br></br>
<h4>Error Rates</h4>
<font face="monospace" color="brown">{cum_err_rate:.2%}</font> cumulative</br>
<font face="monospace" color="green">{theano.config.device}</font>
<h4>Network</h4>
<h5>Specified Parameters:</h5><pre>{ntwk.get_layers_info()}</pre>
<h5>Training Parameters:</h5><pre>{ntwk.get_training_params_info()}</pre>
<h5>Weights:</h5><pre>{ntwk.get_wts_info(detailed=True)}</pre>
<h5>Generated Network:</h5><pre>{ntwk}</pre>
<h4>Legend</h4><pre>sl#) [true_class]: ([wrongs] of [tested] = [error]%)</br>
[false] [image_shown]([rank], [true_probability]% vs [false_probability]%)...
</pre>
<h3>Results</h3>
'''
    filler_main = '\n</br></br><h4 style="color:blue">{}) {} ({} of {} = {:.1%}) </h4>'
    filler_sub = '\n<h4 style="color:red"> {} ({}) </h5>'
    filler_img = '\n<img src="data:image/png;base64,{0}" title="{1}"/> ' \
                 '({2}, {3} vs. {4} )'
    tail = '\n</body></html>'

    ####################### Write HTML
    out_file_name = neural_net_file.replace('.pkl', '.html')
    print('Compiling output html file ', out_file_name, end="\n\n")
    out_file = open(out_file_name, 'w', )
    out_file.write(head)

    check_file_name = neural_net_file.replace('.pkl', '.mistakes.delim')
    check_file = open(check_file_name, 'w')
    print("true,guess,itrue,iguess,path", file=check_file)

    for label in range(1+max(iast2idx.values())):
        # Write Summary
        truech = idx2unic[label]
        error_rate = wrongs[truech] / counts[truech] if counts[truech] > 0 else 0.
        out_file.write(filler_main.format(label, truech, wrongs[truech], counts[truech], error_rate))

        # Write each bad classification
        for falsech, missglyphs in errors[truech].items():
            out_file.write(filler_sub.format(falsech, missglyphs[0].guess_label))

            for glyph in missglyphs:
                img = Image.fromarray((255 * (1 - glyph.pix)).astype(np.uint8))
                buf = io.BytesIO()
                img.save(buf, format='BMP')
                im64 = base64.b64encode(buf.getvalue())

                out_file.write(filler_img.format(im64.decode("ascii"), glyph.in_path,
                                                 glyph.rank_true, glyph.ptruth, glyph.pguess))
                print(truech, falsech, label, glyph.guess_label, glyph.in_path, sep='\t', file=check_file)

    out_file.write(tail)
    out_file.close()
    check_file.close()
    print('Output html file ', out_file_name, end="\n\n")
    print('Check file for culprits', check_file_name, end="\n\n")

###################################### Process all NNfiles


for nnfileorpattern in neural_net_files:
    for nnfile in glob.glob(nnfileorpattern):
        try:
            ret = test_on_network(nnfile)
            html(**ret)
        except:
            print(f"Failed to process {nnfile}")
            print_exc()
