import logging
import sys

from ngramgraph import GramGraph
from post_process import post_process
from scaler import ScalerFactory
from bantry import Bantry, BantryFile
from classifier import Classifier
from ngram import Ngram


def argue(num, default):
    try:
        arg = sys.argv[num]
        assert arg[-4:] == default[-4:]
    except (IndexError, AssertionError):
        arg = default
    return arg

############################## Read Arguments
banti_fname = "sample_images/praasa.box"
nnet_fname = "library/nn.pkl"
scaler_fname = "scalings/relative48.scl"
labels_fname = "labellings/alphacodes.lbl"
ngram_fname = "library/mega.123.pkl"

if len(sys.argv) < 2:
    print("Usage:\n"
          "{} [box={}] [nnet={}] [scaler={}] [labels={}] [ngram={}] -[log=info]"
          "\nloglevel -d:debug, -i:info, -w:warn, -e:error, -c:critical"
          "\nContinuing with defaults"
          "".format(sys.argv[0], banti_fname, nnet_fname,
                    scaler_fname, labels_fname, ngram_fname))

banti_fname = argue(1, banti_fname)
nnet_fname = argue(2, nnet_fname)
scaler_fname = argue(3, scaler_fname)
labels_fname = argue(4, labels_fname)
ngram_fname = argue(5, ngram_fname)

loglevel = {'-c': logging.CRITICAL,
            '-e': logging.ERROR,
            '-w': logging.WARNING,
            '-i': logging.INFO,
            '-d': logging.DEBUG}.get(sys.argv[-1][:2].lower(),
                                     logging.INFO)

replace = lambda s: banti_fname.replace('.box', s)
log_fname = replace('.{}.log'.format(logging._levelToName[loglevel]).lower())
asis_fname = replace('.ml.txt')
nogram_out_fname = replace('.nogram.txt')
ngram_out_fname = replace('.gram.txt')

logging.basicConfig(filename= log_fname,
                    level=loglevel, filemode="w")

############################## Set-up scaler, classifier, ngram etc.
Bantry.scaler = ScalerFactory(scaler_fname)
Bantry.classifier = Classifier(nnet_fname, labels_fname, logbase=1)
ng = Ngram(ngram_fname)
Bantry.ngram = ng
GramGraph.set_ngram(ng)

############################## Read Bantries & get Most likely output
bf = BantryFile(banti_fname)
with open(asis_fname, 'w', encoding='utf-8') as f:
    f.write(post_process(bf.text))

############################## Process using ngrams
ngrammed_lines, notgrammed_lines  = [], []

for linenum in range(bf.num_lines):
    print("Line ", linenum)
    line_bantries = bf.get_line_bantires(linenum)
    gramgraph = GramGraph(line_bantries)
    gramgraph.process_tree()
    notgrammed_lines.append(gramgraph.get_best_apriori_str())
    ngrammed_lines.append(gramgraph.get_best_str())

nogram_out = post_process("\n".join(notgrammed_lines))
with open(nogram_out_fname, 'w', encoding='utf-8') as out_file:
    out_file.write(nogram_out)

ngram_out = post_process("\n".join(ngrammed_lines))
with open(ngram_out_fname, 'w', encoding='utf-8') as out_file:
    out_file.write(ngram_out)

print("Input : ", banti_fname)
print("As is output : ", asis_fname)
print("Without ngram : ", nogram_out_fname)
print("With ngram : ", ngram_out_fname)
print("Log : ", log_fname)