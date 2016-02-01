import sys
import numpy as np

from banti.classifier import Classifier
from banti.scaler import ScalerFactory
from banti.proglyph import ProGlyph
from banti.processedpage import ProcessedPage

banti_file_name = sys.argv[1] if len(sys.argv) > 1 else "sample_images/praasa.box"
nnet_file = sys.argv[2] if len(sys.argv) > 2 else "library/nn.pkl"
scaler_prms_file = sys.argv[3] if len(sys.argv) > 3 else "scalings/relative48.scl"
labellings_file = sys.argv[4] if len(sys.argv) > 4 else "labellings/alphacodes.lbl"

ProGlyph.scaler = ScalerFactory(scaler_prms_file)
ProGlyph.classifier = Classifier(nnet_file, labellings_file, logbase=10, only_top=3)

ppage = ProcessedPage(banti_file_name)

for linenum in range(ppage.num_lines):
    print('*' * 80)
    line_bantries = ppage.get_line_bantires(linenum)
    for bantree in line_bantries:
        print(bantree.scaled)
        for char, logprob in bantree.likelies:
            print(np.exp(logprob), char)

print(ppage.text)