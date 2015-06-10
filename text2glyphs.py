#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Project: Telugu OCR
    Author : Rakeshvara Rao
    License: GNU GPL 3.0

    This module contains functions that take a unicode Telugu string and 
    break it up in to pieces. E.g:- the letter స is broken in to a ✓ and
    a bottom part represented as స again.
"""
import re

VATTU_HAVERS = 'ఖఘఛఝటఢథధఫభ'  # unused
TWO_PIECERS = 'ఘపఫషసహ'
GHO_STYLE = 'TEL'
# This variable is font dependent as ఘో is rendered in two different ways.
# TELugu, KANnada

# aksh_pattern includes space also
aksh_pattern = re.compile(r"([ఁ-ఔృౄ])|( )|(([క-హ]్)*[క-హ][ా-ూె-ౌ])|"
                          r"(([క-హ]్)*[క-హ](?![ా-ూె-్]))|(([క-హ]్)+(?=\s))")


def process_two_piecers(akshara):
    """ Process the aksharas that are written in two pieces"""
    # Class of tick & consonant+vowel 
    if '''ఘాఘుఘూఘౌపుపూఫుఫూషుషూసుసూహా
          హుహూహొహోహౌ'''.find(akshara) >= 0:
        return ['✓', akshara]

    # Class of vowel-mark & underlying consonant base
    if '''ఘిఘీఘెఘేపిపీపెపేఫిఫీఫెఫేషిషీషెషేసిసీసె
          సేహిహీహెహేఘ్ప్ఫ్ష్స్హ్ '''.find(akshara) >= 0:
        return [akshara[1], akshara[0]]

    # Detached ai-karams
    if 'ఘైపైఫైషైసైహై'.find(akshara) >= 0:
        return ['ె', akshara[0], 'ై']

    # gho
    if 'ఘొఘో'.find(akshara) >= 0:
        if GHO_STYLE == 'TEL':  # Telugu style ఘొఘో
            return ['✓', akshara]
        else:  # Kannada style
            return ['ె', 'ఘా' if akshara == 'ఘో' else 'ఘు']

    # Combining marks like saa, pau etc.
    return [akshara]


def process_sans_vattu(akshara):
    """Process one independent symbol or a simple CV pair"""
    glps = []

    # ఏ Special Case
    if akshara == 'ఏ':
        glps += ['ఏ', 'ఎ']

    # Punc, Single Letters
    elif len(akshara) == 1:
        if akshara in TWO_PIECERS:
            glps += ['✓']

        glps += [akshara]

    # Cons + Vowel
    elif len(akshara) == 2:
        if akshara[0] in TWO_PIECERS:
            glps += process_two_piecers(akshara)

        elif akshara[1] in 'ై':
            glps += [akshara[0] + ('ె' if akshara[1] == 'ై' else '')]
            glps += [akshara[1]]

        else:
            glps += [akshara]

    return glps


def process_akshara(akshara):
    """ Processes an Akshara at a time (i.e. syllable by syllable)"""
    aksh_wo_vattulu = akshara[0] + ('' if len(akshara) % 2 else akshara[-1])

    glps = process_sans_vattu(aksh_wo_vattulu)

    for i in range(1, len(akshara) - 1, 2):  # Add each vattu, usually just one
        glps += [akshara[i] + akshara[i + 1]]

    return glps


def process_line(line, pattern=aksh_pattern):
    """The main function of this module; Used to parse one chunk of Telugu text
    """
    glps = []
    for a in pattern.finditer(line):
        glps += process_akshara(a.group())
    return glps


##############################################################################


def main():
    dump = ['ఏతస్మిన్ సిద్ధాశ్రమే దేశే మందాకిన్యా',
            'శైలస్య చిత్రకూటస్య పాదే పూర్వోత్తరే ',
            'ఘోరోఽపేయ పైత్యకారిణ్ లినక్స్ ']
    for line in dump:
        print(line)
        for aks in process_line(line):
            print(aks, end=', ')
        print()

if __name__ == '__main__':
    main()