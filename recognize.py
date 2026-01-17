#!/usr/bin/env python3

############################### Parse Input Arguments #########################
import argparse
import logging


class Formatter(argparse.RawDescriptionHelpFormatter,
                argparse.ArgumentDefaultsHelpFormatter):
    pass

desc = '''Telugu OCR
    Performs OCR on a given image(s) and box/pdf file(s).
      Image files should of sufficient resolution to work well.
      One bit per pixel (binary) TIFF images work best.
      box files are the output of antanci_ocr.
      pdf will be converted to tiff files.
      File matching patterns should be in quotes.
    examples:
         python3 {0} "~/books/andhra*.tif"
         python3 {0} "~/books/*.box"
         python3 {0} ~/books/andhra_maha.pdf
         python3 {0} ~/books/andhra_maha1.tif
         python3 {0} ~/books/andhra_maha1.box
         python3 {0} "~/books/andhra_maha/*"
  '''.format(__file__)

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
prsr.add_argument('--calib', action='store', dest='calibration', type=float,
                  default=1,
                  help='The correction that needs to be applied to the nnet\'s outputs')
prsr.add_argument('--log', action='store', dest='log_level',
                  default='info',
                  help='Level of logging: debug, info, critical etc.')
prsr.add_argument('input_file_or_dir', action='store',
                  help='Can be pdf, tiff, or box file. Or a pattern in quotes.')

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

####################################### Load OCR
from banti.ocr import OCR
print('Initializing the OCR')
recognizer = OCR(args.nnet_fname,
                 args.scaler_fname,
                 args.labels_fname,
                 args.ngram_fname,
                 args.calibration,
                 args.log_level)
print('\t OCR initialized.')


####################################### Actual Code
import glob
from banti.helpers import is_file_of_type, change_ext

def ocr_pattern(pattern):
    for inpt in glob.glob(pattern):
        print("*" * 60)
        print("PROCESSING", inpt)

        if is_file_of_type(inpt, 'pdf'):
            imgs_dir = pdf_to_tiffs(inpt)
            ocr_pattern(imgs_dir + "/*")

        elif is_file_of_type(inpt, 'box'):
            recognizer.ocr_file(inpt)

        elif is_file_of_type(inpt, 'image'):
            recognizer.ocr_file(inpt)

        else:
            print("\tNot an image file.")

ocr_pattern(args.input_file_or_dir)

###################################################
def to_tiff(inpt):
    inptiff = change_ext(inpt, 'converted.tif')
    command = ['convert',
               '-units', 'PixelsPerInch',
               inpt,
               '-compress', 'Group4',
               '-depth', '1',
               '-resample', '400',
               inptiff]
    succ, _, _ = run_command(command, timeout=10)
