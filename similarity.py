import argparse
from itertools import groupby
from music21 import pitch
import numpy as np
import matplotlib.pyplot as plt

import util


def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

def diff(x):
    left, right = x
    if left[0] == 'rest' or right[0] == 'rest':
        # temporary
        return (0, right[1])
    # LOL this is awful
    if left[0] == 'padding' or right[0] == 'padding' or left[0] == 'measure' or right[0] == 'measure' or left[0] == 'end' or right[0] == 'end': 
        return (0, right[1])
    if pitch.Pitch(left[0]) == pitch.Pitch(right[0]):
        diff = 0
    else:
        diff = -1 if pitch.Pitch(left[0]) < pitch.Pitch(right[0]) else 1
    return (diff, right[1])

pdv = util.PitchDurationVocab()

def get_padded_ssm(melody, measure_length, sv):
    ''' |sv| must be a PDV'''
    melody = [sv.i2e[0][i].original for i in melody][1:]
    differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))

    measures = []
    measure_progress = 0
    measure = []
    for i in range(len(differences)):
        if measure_progress == measure_length or i == len(differences)-1:
            measures.append(measure)
            measure = []
            measure_progress = 0
        measure.append(differences[i][0])
        measure_progress += 1

    dist_matrix = np.zeros([len(measures), len(measures)])
    sum_distances = 0
    for i in xrange(len(measures)):
        for j in xrange(len(measures)):
            dist_matrix[i,j] = edit_distance(measures[i], measures[j])
            if i >= j:
               sum_distances += dist_matrix[i,j] 
    max_dist = np.max(dist_matrix)
    ssm = max_dist - dist_matrix 
    return ssm


def get_ssm(f, pdv):
    melody, measure_limit = pdv.mid2orig(f, include_measure_boundaries=False)
    melody = melody[1:-1]

    first_duration = melody[0][1]
    differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))

    measures = []
    measure_progress = 0
    measure = []
    for x in differences:
        if measure_progress >= measure_limit:
            measures.append(measure)
            measure = []
            measure_progress -= measure_limit
        measure.append(x[0])
        measure_progress += x[1]

    dist_matrix = np.zeros([len(measures), len(measures)])
    sum_distances = 0
    for i in xrange(len(measures)):
        for j in xrange(len(measures)):
            dist_matrix[i,j] = float(edit_distance(measures[i], measures[j]))
            if i >= j:
               sum_distances += dist_matrix[i,j] 
    max_dist = np.max(dist_matrix)
    ssm = max_dist - dist_matrix 
    return ssm

    # print measures
    # print sum_distances / ((len(measures)**2)/2)
    # plt.imshow(dist_matrix, cmap='gray', interpolation='nearest')
    # plt.savefig('../similarities/' + args.melody + '.png')


def get_note_sdm(melody, pdv, window):
    """ self-distance matrix """
    ''' |pdv| must be a PDV'''
    melody = [pdv.i2e[0][i].original for i in melody][1:]

    differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))
    rawDiffs = map(lambda x: x[0], differences)

    sdm = np.ones([len(differences), len(differences)]) * 20 # "default" distance 
    for i in xrange(window-1, len(differences)):
        for j in xrange(window-1, len(differences)):
            sdm[i,j] = edit_distance(rawDiffs[i-window:i], rawDiffs[j-window:j])

    return sdm, rawDiffs



