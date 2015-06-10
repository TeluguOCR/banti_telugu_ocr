import ast
import pickle
from theanet.neuralnet import NeuralNet
from bantry import Bantry, BantryFile
from classifier import Classifier
from iast_unicodes import get_index_to_char_converter
from ngram import Ngram
from ngramgraph import GramGraph
from scaler import ScalerFactory


class Machine():
    def __init__(self,
                 box_file,
                 nnet_prms_file,
                 scaler_prms_file,
                 labellings_file,
                 ngram_file):
        Bantry.scaler = ScalerFactory(scaler_prms_file)
        Bantry.classifier = Classifier(nnet_prms_file, labellings_file)
        bf = BantryFile(box_file)
        GramGraph.ngram = Ngram(ngram_file)

        for linenum in range(bf.num_lines):
            line_bantries = bf.get_line_bantires(linenum)
            line_graph = GramGraph(line_bantries)
            line_graph.process_tree()
            line_graph.top_ngram_paths()
