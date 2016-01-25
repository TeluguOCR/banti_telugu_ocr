#!/usr/bin/env python3

############################### Parse Input Arguments #########################
import argparse
import logging


class Formatter(argparse.RawDescriptionHelpFormatter,
                argparse.ArgumentDefaultsHelpFormatter):
    pass

desc = '''Telugu OCR
    Performs OCR on a given tif, box or pdf file. Or a directory with files.
    Tiff file should be 1bpp (i.e. encoded in binary).
    box file is the output of banti_segmenter.
    pdf will be converted to tiff files.
    examples:
         python3 {0} ~/books/andhra_maha.pdf
         python3 {0} ~/books/andhra_maha1.tif
         python3 {0} ~/books/andhra_maha1.box
         python3 {0} ~/books/andhra_maha/'''.format(__file__)

prsr = argparse.ArgumentParser(description=desc, formatter_class=Formatter)

prsr.add_argument('-n', action='store', dest='nnet_fname',
                  default='library/nn.pkl',
                  help='The file where the neural network parmeters are stored')
prsr.add_argument('-s', action='store', dest='scaler_fname',
                  default='scalings/relative48.scl',
                  help='The scaling that should be applied to each image')
prsr.add_argument('-l', action='store', dest='labels_fname',
                  default='labellings/alphacodes.lbl',
                  help='The labels for each Telugu glyph')
prsr.add_argument('-g', action='store', dest='ngram_fname',
                  default='library/mega.123.pkl',
                  help='The nGram dictionaries')
prsr.add_argument('-b', action='store', dest='banti_segmenter',
                  default='./banti_segmenter',
                  help='The binary that convers tiff files to box files.')
prsr.add_argument('--calib', action='store', dest='calibration', type=float,
                  default=1,
                  help='The correction that needs to be applied to '
                       'the nnet\'s outputs')
prsr.add_argument('--log', action='store', dest='log_level',
                  default='info',
                  help='Level of logging: debug, info, critical etc.')
prsr.add_argument('input_file_or_dir', action='store',
                  help='Can be pdf, tiff, or box file. Or a directory full of '
                       'tiff or box files.')

args = prsr.parse_args()

args.log_level = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG}.get(args.log_level.lower(), logging.INFO)

print('Command line Arguments')
for k in sorted(vars(args)):
    print('\t{:20}{}'.format(k, vars(args)[k]))
print()


############################# Helper Functions ################################
import os
import subprocess
from PIL import Image as im

def change_ext(fname, ext):
    name, _ = os.path.splitext(fname)
    if ext[0] != '.':
        ext = '.' + ext

    return name + ext


def is_file_of_type(fname, ext):
    if ext == 'tif':
        checks = '.tif', '.tiff', '.TIF', '.TIFF'

    elif ext == 'box':
        checks = '.box', '.BOX'

    elif ext == 'pdf':
        checks = '.pdf', '.PDF'

    elif ext == 'dir':
        return os.path.isdir(fname)

    else:
        raise ValueError('Unknown extension', ext)

    for e in checks:
        if fname.endswith(e):
            return True
    return False


def run_command(command, timeout=0, prinout=True):
    print('Launched command with timeout={}'
          '\n"{}"'.format(timeout, ' '.join(command)))
    proc = subprocess.Popen(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    success = True

    if timeout:
        try:
            outs, errs = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            success = False

    else:
        outs, errs = proc.communicate()

    if prinout:
        if success:
            print('Success')
            print('STDOUT:\n', outs.decode('utf-8'))
            print('STDERR:\n', errs.decode('utf-8'))
        else:
            print('Failure')
            print('STDOUT:\n', outs.decode('utf-8')[-100:])
            print('STDERR:\n', errs.decode('utf-8')[-100:])

    return success, outs, errs


def pdf_to_tiffs(infile):
    img_dir = os.path.splitext(os.path.abspath(infile))[0]
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_namer = os.path.join(img_dir, 'page_%04d.tif')

    gscommand = ['gs', '-dNOPAUSE',
                 '-q', '-r400', '-sDEVICE=tiffg4', '-dBATCH',
                 #'-dFirstPage=5', '-dLastPage=6',
                 '-dGrayValues=2',
                 '-sOutputFile=' + img_namer,
                 infile]

    run_command(gscommand)
    return img_dir


def tiff_to_box(banti_segmenter, f):
    succ, _, _ = run_command([banti_segmenter, f], timeout=10)

    if succ:
        return change_ext(f, '.box')


def tiff_dir_to_box(img_dir, banti_segmenter):
    print('Converting tiff images in', img_dir, 'to box files.')
    for f in sorted(os.listdir(img_dir)):
        if not is_file_of_type(f, 'tif'):
            continue
        f = os.path.join(img_dir, f)
        if not os.path.exists(change_ext(f, '.box')):
            tiff_to_box(banti_segmenter, f)


####################################### Load OCR
from ocr import OCR

print('Initializing the OCR')
recognizer = OCR(args.nnet_fname,
                 args.scaler_fname,
                 args.labels_fname,
                 args.ngram_fname,
                 args.calibration,
                 args.log_level)
print('Done')


####################################### Helpers

def ocr_box_dir(img_dir):
    print('Recognizing box files in ', img_dir)
    for f in sorted(os.listdir(img_dir)):
        if is_file_of_type(f, 'box'):
            f = os.path.join(img_dir, f)
            print('OCRing', f)
            recognizer.ocr_box_file(f)


def ocr_dir(img_dir):
    tiff_dir_to_box(img_dir, args.banti_segmenter)
    ocr_box_dir(img_dir)

####################################### Actual Code

inpt = args.input_file_or_dir

if is_file_of_type(inpt, 'box'):
    recognizer.ocr_box_file(inpt)

elif is_file_of_type(inpt, 'pdf'):
    imgs_dir = pdf_to_tiffs(inpt)
    ocr_dir(imgs_dir)

elif is_file_of_type(inpt, 'dir'):
    ocr_dir(inpt)

else:
    img = im.open(inpt)
    is_1bpp = img.mode == '1'

    if not is_1bpp:
        inptiff = change_ext(inpt, 'converted.tif')
        command = ['convert',
                   '-units', 'PixelsPerInch',
                   inpt,
                   '-compress', 'Group4',
                   '-depth', '1',
                   '-resample', '400',
                   inptiff]
        succ, _, _ = run_command(command, timeout=30)
    else:
        inptiff = inpt

    box_fname = tiff_to_box(args.banti_segmenter, inptiff)
    if box_fname:
        recognizer.ocr_box_file(box_fname)
    else:
        print("Box file could not be made.")