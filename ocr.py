import logging

from ngramgraph import GramGraph
from post_process import post_process
from scaler import ScalerFactory
from bantry import Bantry, BantryFile
from classifier import Classifier
from ngram import Ngram


class OCR():
    def __init__(self,
                 nnet_fname,
                 scaler_fname,
                 labels_fname,
                 ngram_fname,
                 logbase=1,
                 loglevel=logging.INFO,):
        self.nnet_fname = nnet_fname
        self.scaler_fname = scaler_fname
        self.labels_fname = labels_fname
        self.ngram_fname = ngram_fname
        self.logbase = logbase
        self.loglevel = loglevel
        self.loglevelname = logging._levelToName[loglevel].lower()

        Bantry.scaler = ScalerFactory(scaler_fname)
        Bantry.classifier = Classifier(nnet_fname, labels_fname,
                                       logbase=logbase)
        self.ng = Ngram(ngram_fname)
        Bantry.ngram = self.ng
        GramGraph.set_ngram(self.ng)

    def ocr_box_file(self, box_fname):
        # Set up the names of output files
        replace = lambda s: box_fname.replace('.box', s)

        asis_fname = replace('.ml.txt')
        nogram_out_fname = replace('.nogram.txt')
        ngram_out_fname = replace('.gram.txt')

        log_fname = replace('.{}.log'.format(self.loglevelname))
        logging.basicConfig(filename=log_fname,
                            level=self.loglevel,
                            filemode="w")

        # Read Bantries & get Most likely output
        bf = BantryFile(box_fname)
        with open(asis_fname, 'w', encoding='utf-8') as f:
            f.write(post_process(bf.text))

        # Process using ngrams
        ngrammed_lines, notgrammed_lines = [], []
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

        print("Input : ", box_fname)
        print("As is output : ", asis_fname)
        print("Without ngram : ", nogram_out_fname)
        print("With ngram : ", ngram_out_fname)
        print("Log : ", log_fname)