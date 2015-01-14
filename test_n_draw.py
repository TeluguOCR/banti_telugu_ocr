#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import pickle
import numpy as np
from mallicodes.mallicodes import *
from theanet.neuralnet import NeuralNet
from glyph.gen_training_data import extract_tar
import mallicodes.mallidraw as draw
import theano
import codecs

# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
raise NotImplementedError, "Not a working version, needs to be fixed and " \
                           "tested."

def enc(a):
    return a.encode('utf-8')


def dec(a):
    return a  # a.decode('utf-8')


def unicode2str_decorator(passe_func):
    def call_passe_func(*args, **kwargs):
        newargs = []
        newkwargs = {}
        # Sanitize args and kwargs for unicode
        for a in args:
            if type(a) is unicode:
                newargs.append(enc(a))
            else:
                newargs.append(dec(a))
        for k, v in kwargs.items():
            if type(v) is unicode:
                newkwargs[k] = enc(v)
            else:
                newkwargs[k] = dec(v)
        # Call origional psse function with new, sanitized arguments
        return passe_func(*newargs, **newkwargs)

    return call_passe_func


print = unicode2str_decorator(print)


def printf(s, *args):
    print(enc(s).format(*args))


def format(s, *args):
    return unicode2str_decorator(enc(s).format)(*args)


def join(s, *args):
    return unicode2str_decorator(enc(s).join)(*args)

# #############  Process arguments
if len(sys.argv) < 3:
    printf('Usage {} pickled_params_file imagesdir/tarfile', sys.argv[0])
    sys.exit()

with open(sys.argv[1], 'rb') as prm_pkl_file:
    P = pickle.load(prm_pkl_file)

if os.path.isdir(sys.argv[2]):
    dirs_dir = sys.argv[2]
else:
    tar_name = sys.argv[2]
    dirs_dir = '/tmp/rakesha/1'
    extract_tar(tar_name, dirs_dir)

if dirs_dir[-1] != '/':
    dirs_dir += '/'
home = os.path.abspath(os.path.join(dirs_dir, os.pardir))
output_dir = dirs_dir[:-1] + '_out/'


def mkdir(direc):
    print('Making dir ', direc, end=' ')
    try:
        os.mkdir(direc)
        print('Made')
    except OSError as exc:
        if os.path.isdir(direc):
            print('Exits')
        else:
            raise exc


mkdir(output_dir)

css_loc = os.path.join(output_dir, 'collapse.css')
printf('Home: {} \nInput:{} \nOutput:{} \nCSS:{}', home, dirs_dir, output_dir,
       css_loc)
with open(css_loc, 'w') as f:
    f.write('.FAQ { \n'
            '    vertical-align: top; \n'
            '    height:auto !important; \n'
            '}\n'
            '.list {\n'
            '    display:none; \n'
            '    height:auto;\n'
            '    margin:0;\n'
            '    float: left;\n'
            '}\n'
            '.show {\n'
            '    display: none; \n'
            '}\n'
            '.hide:target + .show {\n'
            '    display: inline; \n'
            '}\n'
            '.hide:target {\n'
            '    display: none; \n'
            '}\n'
            '.hide:target ~ .list {\n'
            '    display:inline; \n'
            '}\n'
            '\n'
            '/*style the (+) and (-) */\n'
            '.hide, .show {\n'
            '    width: 30px;\n'
            '    height: 30px;\n'
            '    border-radius: 30px;\n'
            '    font-size: 20px;\n'
            '    color: #fff;\n'
            '    text-shadow: 0 1px 0 #666;\n'
            '    text-align: center;\n'
            '    text-decoration: none;\n'
            '    box-shadow: 1px 1px 2px #000;\n'
            '    background: #cccbbb;\n'
            '    opacity: .95;\n'
            '    margin-right: 0;\n'
            '    float: left;\n'
            '    margin-bottom: 25px;\n'
            '}\n'
            '\n'
            '.hide:hover, .show:hover {\n'
            '    color: #eee;\n'
            '    text-shadow: 0 0 1px #666;\n'
            '    text-decoration: none;\n'
            '    box-shadow: 0 0 4px #222 inset;\n'
            '    opacity: 1;\n'
            '    margin-bottom: 25px;\n'
            '}\n\n'
            '.list p{\n'
            '    height:auto;\n'
            '    margin:0;\n'
            '}\n'
            '.question {\n'
            '    float: left;\n'
            '    height: auto;\n'
            '    width: 90%;\n'
            '    line-height: 20px;\n'
            '    padding-left: 20px;\n'
            '    margin-bottom: 25px;\n'
            '    font-style: italic;\n'
            '}')


def get_new_dir(orig_dir):
    return os.path.join(output_dir, os.path.relpath(orig_dir, dirs_dir))


def get_old_dir(orig_dir, new_dir):
    return os.path.join(os.path.relpath(orig_dir, new_dir))

################## CNN
BATCH_SZ = 1
P['BATCH_SZ'] = BATCH_SZ

y_data = np.empty(BATCH_SZ, dtype='int32')
x_data = np.empty((BATCH_SZ, P['IMG_SZ'] ** 2), dtype='float')

sh_x = theano.shared(np.asarray(x_data, theano.config.floatX), borrow=True)
sh_y = theano.shared(np.asarray(y_data, 'int32'), borrow=True)


def update_shared():
    sh_x.set_value(x_data)
    sh_y.set_value(y_data)


ntwk = NeuralNet(**P)
model = ntwk.get_test_model(sh_x, sh_y)

##################

class Index():
    def __init__(self, dir_path):
        self.loc = dir_path
        self.good = []
        self.bad = []
        self.dirs = [('UP', '..')]

    def add_good(self, i):
        self.good.append(i)

    def add_bad(self, i):
        self.bad.append(i)

    def add_subdir(self, i):
        self.dirs.append(i)

    def close(self):
        def listtotable(ls):
            return join(u'\n',
                        [format(u'<img src="{}"><img src="{}">', out, inp)
                         for inp, out in ls])

        def subdir_encase(i, o):
            return format('<a href="{1}/index.html">{0}({1})</a>', i, o)

        subdir_text = join('\n', [subdir_encase(*i) for i in self.dirs])

        enc, wrng = len(self.good), len(self.bad)
        total = enc + wrng
        try:
            summary = format('Total: {} Correct:{}({:.2f}%) Wrong:{}({:.2f}%)',
                             total, enc, 100 * float(enc) / total, wrng,
                             100 * float(wrng) / total)
        except ZeroDivisionError:
            summary = 'Characters'

        text = format(u'''{}<div class="FAQ">
            <a href="#hide1" class="hide" id="hide1">+</a>
            <a href="#show1" class="show" id="show1">-</a>
            <div class="question">Right</div>
                <div class="list">
                {}
                </div>
        </div>
        </br>
        <div class="FAQ">
            <a href="#hide2" class="hide" id="hide2">+</a>
            <a href="#show2" class="show" id="show2">-</a>
            <div class="question">Wrong</div>
                <div class="list">
                {}
                </div>
        </div>''', summary, listtotable(self.good), listtotable(self.bad))

        print('Writing ', os.path.join(self.loc, 'index.html'))
        f = codecs.open(os.path.join(self.loc, 'index.html'), 'w', 'utf-8')

        content = format(u'''<html xml:lang="te" xmlns="http://www.w3.org/1999/xhtml" lang="te">
<head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>{}</title>
<link rel="stylesheet" href="{}" type="text/css">
</head>
<body>
{}
</body>
        ''', os.path.basename(self.loc),
                         os.path.relpath(css_loc, self.loc),
                         subdir_text + text)
        f.write(content.decode('utf-8'))
        f.close()

##################
idir = 0
for dirpath, dirnames, filenames in os.walk(dirs_dir):
    # Make the required directory to output
    new_dir_path = get_new_dir(dirpath)
    mkdir(new_dir_path)

    # Init html file
    html = Index(new_dir_path)

    # Add sub-directories
    for dirname in sorted(dirnames):
        uni = unicodes[
            char_indices[dirname]] if dirname in char_indices else dirname
        print(uni, dirname)
        html.add_subdir((uni, dirname))

    # Get unicode
    glyph = os.path.basename(dirpath)
    idir += 1
    try:
        mallidx = char_indices[glyph]
        printf('### {}\nProcessing images for {}', idir, glyph)
        print(unicodes[mallidx])
    except KeyError:
        print("Could not find '", glyph, "' in mallicodes")
        html.close()
        continue

    for filename in filenames:
        if filename[-4:] != '.png':
            print("Skipping non png file", filename)
            continue
        file_path = os.path.join(dirpath, filename)
        font, style, ID, dtbpairs = gen.SplitFileName(filename)

        # Open image and process
        imod = 0
        for x_img in gen.GetScaledImgVectors(file_path):
            imod += 1
            y_data[0] = mallidx
            x_data[0] = x_img
            update_shared()
            bits, preds, truth = model(0)
            ifname = filename + str(imod)
            ifname_full = os.path.join(new_dir_path, ifname)
            draw.draw_save(bits[0].tolist(), unicodes[preds[0]], ifname_full)
            names = (ifname + '.jpg', get_old_dir(file_path, new_dir_path))
            if preds[0] == truth[0]:
                html.add_good(names)
            else:
                print(unicodes[preds[0]], end='\t')
                html.add_bad(names)
    print()
    html.close()
    #sys.exit()

