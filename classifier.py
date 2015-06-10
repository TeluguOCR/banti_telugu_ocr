import pickle
import numpy as np
from theanet.neuralnet import NeuralNet
from iast_unicodes import get_index_to_char_converter


class Classifier():
    def __init__(self, nnet_prms_file, labellings_file, logbase=1, only_top=5):
        with open(nnet_prms_file, 'rb') as nnet_prms_fp:
            nnet_prms = pickle.load(nnet_prms_fp)

        nnet_prms['training_params']['BATCH_SZ'] = 1
        self.ntwk = NeuralNet(**nnet_prms)
        self.tester = self.ntwk.get_data_test_model()
        self.ht = nnet_prms['layers'][0][1]['img_sz']
        self.logbase = logbase
        self.only_top = only_top

        idx2char = get_index_to_char_converter(labellings_file)
        nclasses = nnet_prms['layers'][-1][1]["n_out"]
        self.chars = np.array([idx2char(i) for i in range(nclasses)])

    def __call__(self, scaled_glp):
        img = scaled_glp.pix.astype('float32').reshape((1, self.ht, self.ht))

        if self.ntwk.takes_aux():
            dtopbot = scaled_glp.dtop, scaled_glp.dbot
            aux_data = np.array([[dtopbot, dtopbot]], dtype='float32')
            logprobs, preds = self.tester(img, aux_data)
        else:
            logprobs, preds = self.tester(img)

        logprobs = logprobs[0]

        if self.only_top:
            decent = np.argpartition(logprobs, -self.only_top)[-self.only_top:]
            chars = self.chars[decent]
            logprobs = logprobs[decent]
        else:
            chars = self.chars

        return chars, logprobs/self.logbase

if __name__ == "__main__":
    import sys
    from scaler import ScalerFactory
    from bantry import Bantry, BantryFile

    banti_file_name = sys.argv[1]
    nnet_file = sys.argv[2]
    scaler_prms_file = sys.argv[3] if len(sys.argv) > 3 else "scalings/relative48.scl"
    labellings_file = sys.argv[4] if len(sys.argv) > 4 else "labelings/alphacodes.lbl"

    Bantry.scaler = ScalerFactory(scaler_prms_file)
    Bantry.classifier = Classifier(nnet_file, labellings_file, logbase=10, only_top=3)

    bf = BantryFile(banti_file_name)

    for linenum in range(bf.num_lines):
        print('*' * 80)
        line_bantries = bf.get_line_bantires(linenum)
        for bantree in line_bantries:
            print(bantree.scaled)
            for char, logprob in zip(*bantree.likelies):
                print(np.exp(logprob), char)