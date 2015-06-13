import pickle
import numpy as np
from theanet.neuralnet import NeuralNet
from iast_unicodes import LabelToUnicodeConverter
import logging
logger = logging.getLogger(__name__)
logi = logger.info
logd = logger.debug


class Classifier():
    def __init__(self, nnet_prms_file, labellings_file, logbase=2, only_top=5):
        with open(nnet_prms_file, 'rb') as nnet_prms_fp:
            nnet_prms = pickle.load(nnet_prms_fp)

        nnet_prms['training_params']['BATCH_SZ'] = 1
        self.ntwk = NeuralNet(**nnet_prms)
        self.tester = self.ntwk.get_data_test_model()
        self.ht = nnet_prms['layers'][0][1]['img_sz']
        self.logbase = logbase
        self.only_top = only_top

        self.unichars = LabelToUnicodeConverter(labellings_file)
        self.nclasses = nnet_prms['layers'][-1][1]["n_out"]

        logi("Network {}".format(self.ntwk))
        logi("LogBase {}".format(self.logbase))
        logi("OnlyTop {}".format(self.only_top))

    def __call__(self, scaled_glp):
        img = scaled_glp.pix.astype('float32').reshape((1, self.ht, self.ht))

        if self.ntwk.takes_aux():
            dtopbot = scaled_glp.dtop, scaled_glp.dbot
            aux_data = np.array([[dtopbot, dtopbot]], dtype='float32')
            logprobs, preds = self.tester(img, aux_data)
        else:
            logprobs, preds = self.tester(img)

        logprobs = logprobs[0]/self.logbase

        if self.only_top:
            decent = np.argpartition(logprobs, -self.only_top)[-self.only_top:]
            if logger.isEnabledFor(logging.INFO):
                decent = decent[np.argsort(-logprobs[decent])]
        else:
            decent = np.arange(self.nclasses)

        return [(ch, logprobs[i])
                for i in decent
                for ch in self.unichars[i]]

if __name__ == "__main__":
    import sys
    from scaler import ScalerFactory
    from bantry import Bantry, BantryFile

    banti_file_name = sys.argv[1] if len(sys.argv) > 1 else "sample_images/praasa.box"
    nnet_file = sys.argv[2] if len(sys.argv) > 2 else "library/nn.pkl"
    scaler_prms_file = sys.argv[3] if len(sys.argv) > 3 else "scalings/relative48.scl"
    labellings_file = sys.argv[4] if len(sys.argv) > 4 else "labellings/alphacodes.lbl"

    Bantry.scaler = ScalerFactory(scaler_prms_file)
    Bantry.classifier = Classifier(nnet_file, labellings_file, logbase=10, only_top=3)

    bf = BantryFile(banti_file_name)

    for linenum in range(bf.num_lines):
        print('*' * 80)
        line_bantries = bf.get_line_bantires(linenum)
        for bantree in line_bantries:
            print(bantree.scaled)
            for char, logprob in bantree.likelies:
                print(np.exp(logprob), char)

    print(bf.text)