#!/usr/bin/python

from __future__ import print_function
import os, sys, cPickle
import numpy    as     np
from Mallicodes import *
from   CnnLayers   import Cnn
import gen_cnn_data as gen
import theano
import time
from   collections import defaultdict, Counter

# Pothana Batch Size vs time
# 20000 => 218 | 1000 => 172 | 10-100 => 160 | 1 => 202
BATCH_SZ = 100
P = {}
with open(sys.argv[1], 'rb') as prm_pkl_file:
    P.update(cPickle.load(prm_pkl_file))
P['BATCH_SZ'] = BATCH_SZ

if len(sys.argv) > 2: tar_name = sys.argv[2]  
else:                   tar_name = 'archive/training_data.tar.gz'
dirs_dir = '/tmp/rakesha/'

gen.ExtractTarFile(tar_name, dirs_dir)

class meta:
    def __init__(self, font, style, ID):
        self.font, self.style, self.ID = font, style, ID
    def __repr__(self):
        return ' '.join((self.font, self.style, self.ID))

m_data  = [meta('Dummy', 'DM', '0000000') for i in range(BATCH_SZ)]
y_data =  np.empty(BATCH_SZ, dtype='int32')
x_data  = np.empty((BATCH_SZ, P['IMG_SZ']**2), dtype='float')
def GetImgBatch():
    global m_data, x_data, y_data
    isample, idir = 0, 0

    for dirpath, dirnames, filenames in os.walk(dirs_dir):
        glyph = os.path.basename(dirpath)
        idir += 1
        print(idir, 'Processing images for ', glyph,)
        try:    
            malli = char_indices[glyph]
        except KeyError: 
            print("Could not find '", glyph, "' in mallicodes")
            continue
        
        for filename in filenames:
            if filename[-4:] != '.tif':
                print( "Skipping non tiff file", filename)
                continue
            file_path = os.path.join(dirpath, filename)
            font, style, ID, dtbpairs = gen.SplitFileName(filename)

            # Open imgage and process
            for x_img in gen.GetScaledImgVectors(file_path):
                y_data[isample] = malli
                x_data[isample] = x_img
                m_data[isample] = meta(font, style, ID)
                isample += 1
                if isample == BATCH_SZ:
                    yield isample #(x_data, y_data, m_data)
                    isample = 0
    yield isample

N = len(unicodes)
wrong_bits = np.zeros_like(mallicodes[0])
errs = np.zeros((N, N), dtype='int')
counts = np.zeros(N, dtype='int')
def ProcesResults(results, n):
    bits, preds, truth = results
    for i in range(n):
        t = truth[i]
        p = preds[i]
        m = m_data[i]
        counts[t] += 1
        if t != p:
            wrong_bits[bits[i] != mallicodes[t]] += 1
            errs[t][p] +=1

sh_x = theano.shared(np.asarray(x_data, theano.config.floatX), borrow=True)
sh_y = theano.shared(np.asarray(y_data, 'int32'), borrow=True)

def update_shared():
    global sh_x, sh_y, sh_m
    sh_x.set_value(x_data)
    sh_y.set_value(y_data)

ntwk = Cnn(P)
model = ntwk.get_full_test_model(sh_x, sh_y)

start = time.clock()
for n in GetImgBatch():
    update_shared()
    ProcesResults(model(0), n)

total = counts.sum()
wrong = errs.sum()
rwrong = errs.sum(1)
print("right {0}/{2} = {3:2.2f}%\nwrong {1}/{2} = {4:2.2f}%".format(
        total-wrong, wrong, total, 100 - float(100*wrong)/total, 
        float(100*wrong)/total))
print('Batch Size :', BATCH_SZ, 'Time :', time.clock()-start, 'secs')

def print_defaultcounter(err_dict):
    for truth in err_dict:
        tot = sum(err_dict[truth].values())
        print('\n{} ({}%):'.format(truth, 100*tot/counts[truth]), end=' ')
        for guess, count in sorted(err_dict[truth].items(), key=lambda x:x[1], reverse=True):
            print(guess, '(', 100*count/counts[truth], ')', sep='', end=' ')

def pprint_mat(err_mat):
    for i in np.arange(N)[rwrong > 0]:
        row = err_mat[i]
        print('\n{} ({}):'.format(unicodes[i], 100*rwrong[i]/counts[i]), end=' ')
        for j in np.argsort(-row):
            if row[j] == 0: break
            print(unicodes[j], '(', 100*row[j]/counts[i], ')', sep='', end=' ')

#print_defaultcounter(errs)
pprint_mat(errs)
print()
print('% Wrong', 10000*wrong_bits/total)