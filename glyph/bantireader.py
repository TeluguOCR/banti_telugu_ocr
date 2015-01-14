# -*- coding: utf-8 -*-
from __future__ import print_function
from glyph import Glyph
from scaler import ScalerFactory


class BantiReader(object):
    def __init__(self, filename, scaler_params, batch_sz=20):
        with open(filename) as banti_fp:
            self.banti_file = banti_fp.read()
        self.scaler = ScalerFactory(scaler_params)
        self.batch_sz = batch_sz
        self.params = scaler_params

    def __call__(self):
        ret_data = []
        ret_meta = []
        isample = 0
        for glp_entry in self.banti_file.split('\n'):
            glp = Glyph.fromSixPack(glp_entry)
            if glp is None:
                print('Malformed entry : ', glp_entry)
                continue

            scaled_glp = self.scaler(glp)
            ret_data.append(scaled_glp)
            ret_meta.append((glp.linenum, glp.wordnum, glp.dtop, glp.dbot))
            isample += 1
            if isample == self.batch_sz:
                yield (isample, ret_meta, ret_data)
                isample = 0
                ret_meta = []
                ret_data = []

        yield (isample, ret_meta, ret_data)

###############################################################################


if __name__ == "__main__":
    import ast
    import sys

    banti_file_name = sys.argv[1]
    scaler_params_file = sys.argv[2]
    with open(scaler_params_file) as f:
        scaler_params = ast.literal_eval(f.read())

    gg = BantiReader(banti_file_name, scaler_params)

    iimg = 0
    linenum = 0
    for nsamples, meta, data in gg():
        for m, scaled_glp in zip(meta, data):
            if m[0] > linenum:
                linenum = m[0]
                print('*' * 60)
            print(scaled_glp)
            print('Img : {} DTOP:{} to {:.2f}  DBOT:{} to {:.2f}'.format(
                iimg, int(m[2]), scaled_glp.dtop, int(m[3]), scaled_glp.dbot))
            iimg += 1