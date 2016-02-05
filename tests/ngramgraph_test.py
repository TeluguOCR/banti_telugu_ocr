import sys
from banti.classifier import Classifier
from banti.ngram import Ngram
from banti.ngramgraph import GramGraph
from banti.proglyph import ProGlyph
from banti.scaler import ScalerFactory
from banti.processedpage import ProcessedPage

nnet_file = sys.argv[1] if len(sys.argv) > 1 else "library/nn.pkl"
input_file_name = sys.argv[2] if len(sys.argv) > 2 else "sample_images/praasa.box"
scaler_prms_file = sys.argv[3] if len(sys.argv) > 3 else "scalings/relative48.scl"
labellings_file = sys.argv[4] if len(sys.argv) > 4 else "labellings/alphacodes.lbl"
ngram_file = "library/mega.123.pkl"

ProGlyph.scaler = ScalerFactory(scaler_prms_file)
ProGlyph.classifier = Classifier(nnet_file, labellings_file, logbase=1)
ngram = Ngram(ngram_file)
ProGlyph.ngram = ngram
GramGraph.set_ngram(ngram)

procd_page = ProcessedPage(input_file_name)


for linenum in range(procd_page.num_lines):
    print('*' * 80)
    glyphs = procd_page.get_line_glyphs(linenum)
    gramgraph = GramGraph(glyphs)
    gramgraph.process_tree()
    gramgraph.find_top_ngram_paths()
    for node, children in enumerate(gramgraph.lchildren):
        print(gramgraph.top_pathnodes_at(node, 1))
    print(gramgraph.get_best_str('|'))
    print(gramgraph.get_best_apriori_str('|'))
