#! /usr/bin/python
# -*- coding: utf-8 -*-
import ast
import bz2
import json
import os
import re
import sys
import tarfile
from random import choice
from collections import defaultdict

import numpy as np
from PIL import Image

from glyph import BasicGlyph
from scaler import ScalerFactory



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

            img = Image.open(os.path.join(dir_path, file_path))
            font, style, id_, dtbpairs = split_file_name(file_path)
            glyphs[(font, style)].append(BasicGlyph((img, dtbpairs)))

        self.glyphs = glyphs
        self.dir = dir_path
        self.name = os.path.basename(dir_path)
        self.scaler = default_scaler
        self.default_getter = default_sampler

    def get_all(self, ):
        for font_style, glyph_list in list(self.glyphs.items()):
            for glyph in glyph_list:
                for dtop, dbot in glyph.dtopbot_pairs:
                    yield (self.scaler(BasicGlyph((glyph.img, dtop, dbot))),
                           font_style)

    def get_one_per_file(self, ):
        for font_style, glyph_list in list(self.glyphs.items()):
            for glyph in glyph_list:
                dtop, dbot = choice(glyph.dtopbot_pairs)
                yield (self.scaler(BasicGlyph((glyph.img, dtop, dbot))),
                       font_style)

    def get_one_per_style(self, ):
        for font_style, glyph_list in list(self.glyphs.items()):
            yield (self.scaler(choice(glyph_list)), font_style)

    def get_relative(self, ):
        for font_style, glyph_list in list(self.glyphs.items()):
            scaled_glyph = None
            scaled_dtopbots = []

            # Find scaled dtop, dbot for all imgs (very very inefficient)
            for glyph in glyph_list:
                for dtop, dbot in glyph.dtopbot_pairs:
                    scaled_glyph = self.scaler(BasicGlyph((glyph.img, dtop, dbot)))
                    scaled_dtopbots.append((scaled_glyph.dtop,
                                            scaled_glyph.dbot))

            # Now find max min
            highs = lowws = scaled_dtopbots[0]
            for pair in scaled_dtopbots:
                if sum(pair) > sum(highs):
                    highs = pair
                if sum(pair) < sum(lowws):
                    lowws = pair

            yield scaled_glyph, \
                  font_style, \
                  np.round(np.array((highs, lowws)), 3)

    def getter(self):
        return getattr(self, self.default_getter)()

################################# Build Matrices ############################


def process_root_dir(root_dir, codes, image_sampler, scaler):
    """


    :param root_dir:
    :param codes:
    :param image_sampler:
    :param scaler:
    :type scaler: scaler.Relative
    :return:
    """
    lbl_data, img_data, aux_data, meta_data = [], [], [], []
    idir = 0

    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path):
            print('Skipping non-directory ', dir_path)
            continue

        idir += 1
        print(idir, 'Processing Dir :', dir_path)

        glp_dir = GlyphDir(dir_path, scaler, image_sampler)
        klass = codes[glp_dir.name]

        for data in glp_dir.getter():
            lbl_data.append(klass)
            img_data.append(data[0].pix)
            meta_data.append(data[1] + (glp_dir.name,))

            if len(data) > 2:
                aux_data.append(data[2])

    return lbl_data, img_data, meta_data, aux_data


################################# ZIP UNZIP #################################


def extract_tar(tar_name, target_dir):
    print("Extracting the tarfile", tar_name, "to", target_dir)
    data_tar = tarfile.open(tar_name)
    data_tar.extractall(path=target_dir)
    data_tar.close()


def save_bz2(file_name, variable):
    if not file_name.endswith('.bz2'):
        file_name += '.bz2'

    print("Saving ", file_name)
    bz2_file = bz2.BZ2File(file_name, 'wb')
    bz2_file.write(json.dumps(variable).encode('utf-8'))
    bz2_file.close()


################################# MAIN #################################

# https://www.dropbox.com/s/0psgcha3l47dl21/training_data.tar.gz
if len(sys.argv) < 6:
    print('''Usage:
{0} <images.tar.gz/dir> <scaler.scl> <codes.lbl> <image_sampler> <prefix>

    images.tar.gz/dir : contains the samples images in their directories,
        can be a tar file or an extracted directory.
    scaler.scl : The scaler paramerters dict in an ast readable file
    codes.lbl: map from glyph names to integers
    image_sampler: All, One per file, One per style, relative etc.
    prefix   : <prefix>.x.bz2 and <prefix>.y.bz2 etc. will be saved

e.g.s:-
{0} images.tar.gz absolute64.scl alphacodes.lbl get_one_per_style abs68
{0} /tmp/tel_ocr/numbers absolute09.scl numbers.lbl get_all num32
{0} images.tar.gz relative48.scl alphacodes.lbl get_relative rel48
{0} /tmp/tel_ocr absolute64.scl mallicodes.lbl get_one_per_file abs68malli
    '''.format(sys.argv[0]))
    sys.exit(-1)

######################################## Process Arguments
_, tar_file, scaler_prms_file, codes_file, image_sampler, prefix = sys.argv
dirs_dir = '/tmp/rakesha/'


######################################## Open parameter files
with open(scaler_prms_file) as scaler_fp:
    scaler_params = ast.literal_eval(scaler_fp.read())
    scaler = ScalerFactory(scaler_params)

with open(codes_file) as codes_fp:
    codes = ast.literal_eval(codes_fp.read())


################################### Extract tar and change to relevant sub-dir
if os.path.isdir(tar_file):
    dirs_dir = tar_file
else:
    extract_tar(tar_file, dirs_dir)

if len(os.listdir(dirs_dir)) == 1:
    dirs_dir = os.path.join(dirs_dir, os.listdir(dirs_dir)[0])
    assert os.path.isdir(dirs_dir)


################################### Actual Processing
print("Processing root directory ", dirs_dir)
labels, imgs, meta, aux = process_root_dir(dirs_dir, codes,
                                           image_sampler, scaler)

################################### Apply a random permutation
n_samples = len(labels)
print("Found a total of", n_samples, "samples")

perm = np.random.permutation(len(labels))
permute = lambda lst: np.asarray(lst)[perm].tolist()
labels = permute(labels)
imgs = permute(imgs)
meta = permute(meta)
if aux:
    aux = permute(aux)

################################### Save
save_bz2(prefix+'.labels', labels)
save_bz2(prefix+'.lines', aux)
save_bz2(prefix+'.meta', meta)
save_bz2(prefix+'.images', imgs)

with open(prefix+'.scl', 'w') as scaler_fp:
    print("Saving", prefix+'.scl')
    json.dump(scaler_params, scaler_fp)