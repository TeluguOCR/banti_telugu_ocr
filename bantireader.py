# -*- coding: utf-8 -*-
from _bantry import BoxFileReader
from glyph import Glyph
from scaler import ScalerFactory


###############################################################################


if __name__ == "__main__":
    import ast
    import sys

    banti_file_name = sys.argv[1]
    scaler_params_file = sys.argv[2]
    with open(scaler_params_file) as f:
        scaler_params = ast.literal_eval(f.read())

    gg = BoxFileReader(banti_file_name, scaler_params)

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