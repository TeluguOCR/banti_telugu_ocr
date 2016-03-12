import logging
from .helpers import get_ext_changer
from .helpers import default_nnet_fname, default_scaler_fname
from .helpers import default_labels_fname, default_ngram_fname

from .classifier import Classifier
from .ngram import Ngram
from .ngramgraph import GramGraph
from .post_process import post_process
from .proglyph import ProGlyph
from .scaler import ScalerFactory
from .processedpage import ProcessedPage


class OCR():
    def __init__(self,
                 nnet_fname = default_nnet_fname(),
                 scaler_fname = default_scaler_fname(),
                 labels_fname = default_labels_fname(),
                 ngram_fname = default_ngram_fname(),
                 logbase=1,
                 loglevel=logging.INFO,):
        self.nnet_fname = nnet_fname
        self.scaler_fname = scaler_fname
        self.labels_fname = labels_fname
        self.ngram_fname = ngram_fname
        self.logbase = logbase
        self.loglevel = loglevel
        self.loglevelname = logging._levelToName[loglevel].lower()

        ProGlyph.scaler = ScalerFactory(scaler_fname)
        ProGlyph.classifier = Classifier(nnet_fname, labels_fname, logbase=logbase)
        ProGlyph.ngram = self.ng =Ngram(ngram_fname)

        GramGraph.set_ngram(self.ng)

    def ocr_file(self, input_fname):
        # Set up the names of output files
        change_ext = get_ext_changer(input_fname)

        asis_fname = change_ext('.ml.txt')
        nogram_out_fname = change_ext('.nogram.txt')
        ngram_out_fname = change_ext('.gram.txt')

        log_fname = change_ext('.{}.log'.format(self.loglevelname))
        logging.basicConfig(filename=log_fname,
                            level=self.loglevel,
                            filemode="w")

        # Read Processed Glyphs & get Most likely output
        print("Classifing glyphs...")
        procd_page = ProcessedPage(input_fname)
        with open(asis_fname, 'w', encoding='utf-8') as f:
            f.write(post_process(procd_page.text))

        # Process using ngrams
        print("Finding most likely sentences...")
        ngrammed_lines, notgrammed_lines = [], []
        for linenum in range(procd_page.num_lines):
            print("Line ", linenum)
            line_pglyphs = procd_page.get_line_glyphs(linenum)
            gramgraph = GramGraph(line_pglyphs)
            gramgraph.process_tree()
            notgrammed_lines.append(gramgraph.get_best_apriori_str())
            ngrammed_lines.append(gramgraph.get_best_str())

        nogram_out = post_process("\n".join(notgrammed_lines))
        with open(nogram_out_fname, 'w', encoding='utf-8') as out_file:
            out_file.write(nogram_out)

        ngram_out = post_process("\n".join(ngrammed_lines))
        with open(ngram_out_fname, 'w', encoding='utf-8') as out_file:
            out_file.write(ngram_out)

        print("Input : ", input_fname)
        print("As is output : ", asis_fname)
        print("Without ngram : ", nogram_out_fname)
        print("With ngram : ", ngram_out_fname)
        print("Log : ", log_fname)
        
        return ngram_out
