import sys
from banti.classifier import Classifier
from banti.ngram import Ngram
from banti.ngramgraph import GramGraph
from banti.proglyph import ProGlyph
from banti.scaler import ScalerFactory
from banti.processedpage import ProcessedPage

nnet_file = sys.argv[1] if len(sys.argv) > 1 else "library/nn.pkl"
banti_file_name = sys.argv[2] if len(sys.argv) > 2 else "sample_images/praasa.box"
scaler_prms_file = sys.argv[3] if len(sys.argv) > 3 else "scalings/relative48.scl"
labellings_file = sys.argv[4] if len(sys.argv) > 4 else "labellings/alphacodes.lbl"
ngram_file = "library/mega.123.pkl"

ProGlyph.scaler = ScalerFactory(scaler_prms_file)
ProGlyph.classifier = Classifier(nnet_file, labellings_file, logbase=1)
bf = ProcessedPage(banti_file_name)

ngram = Ngram(ngram_file)
GramGraph.set_ngram(ngram)

for linenum in range(bf.num_lines):
    print('*' * 80)
    bantires = bf.get_line_bantires(linenum)
    gramgraph = GramGraph(bantires)
    gramgraph.process_tree()
    gramgraph.find_top_ngram_paths()
    for node, children in enumerate(gramgraph.lchildren):
        print(gramgraph.top_pathnodes_at(node, 1))
    print(gramgraph.get_best_str('|'))
    print(gramgraph.get_best_apriori_str('|'))
