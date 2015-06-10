import ast
import pickle
from theanet.neuralnet import NeuralNet
from bantireader import BoxFileReader
from iast_unicodes import get_index_to_char_converter


class Machine():
    def __init__(self,
                 box_file,
                 nnet_prms_file,
                 scaler_prms_file,
                 codes_file,
                 labelings_file,
                 ngram_file):

        ############################################# Load Params

        with open(scaler_prms_file, 'r') as scaler_prms_fp:
            scaler_prms = ast.literal_eval(scaler_prms_fp.read())

        with open(nnet_prms_file, 'rb') as nnet_prms_fp:
            nnet_prms = pickle.load(nnet_prms_fp)

        with open(labelings_file, encoding='utf-8') as labels_fp:
            labellings = ast.literal_eval(labels_fp.read())

        # print(labellings)
        self.index_to_char = get_index_to_char_converter(labellings)

        self.gg = BoxFileReader(box_file, scaler_prms)

        ############################################# Init Network
        ht = gg.scaler.params.TOTHT
        nnet_prms['training_params']['BATCH_SZ'] = 1
        nnet_prms['layers'][0][1]['img_sz'] = ht
        self.ntwk = NeuralNet(**nnet_prms)
        self.tester = self.ntwk.get_data_test_model()



