import logging
import sys
from ngramgraph import GramGraph
from post_process import post_process
from scaler import ScalerFactory
from bantry import Bantry, BantryFile
from classifier import Classifier
from ngram import Ngram

############################## Read Arguments
banti_file_name = sys.argv[1]
nnet_file = sys.argv[2] if len(sys.argv) > 2 else "library/nn.pkl"
scaler_prms_file = sys.argv[3] if len(sys.argv) > 3 else "scalings/relative48.scl"
labellings_file = sys.argv[4] if len(sys.argv) > 4 else "labellings/alphacodes.lbl"
ngram_file = "library/mega.123.pkl"

logging.basicConfig(filename=banti_file_name.replace('.box', '.log'),
                    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
                    filemode="w")

############################## Set-up scaler, classifier, ngram etc.
Bantry.scaler = ScalerFactory(scaler_prms_file)
Bantry.classifier = Classifier(nnet_file, labellings_file, logbase=1)
GramGraph.set_ngram(Ngram(ngram_file))

############################## Read Bantries & get Most likely output
bf = BantryFile(banti_file_name)
most_likely_file_name = banti_file_name.replace('.box', '.ml.txt')
with open(most_likely_file_name, 'w', encoding='utf-8') as f:
    f.write(post_process(bf.text))

############################## Process using ngrams
best_ngrammed_lines = []

for linenum in range(bf.num_lines):
    line_bantries = bf.get_line_bantires(linenum)
    gramgraph = GramGraph(line_bantries)
    gramgraph.process_tree()
    best_ngrammed_lines.append(gramgraph.best_str)

ngram_out = post_process("\n".join(best_ngrammed_lines))
out_file_name = banti_file_name.replace('.box', '.gram.txt')
with open(out_file_name, 'w', encoding='utf-8') as out_file:
    out_file.write(post_process(ngram_out))