import argparse
from itertools import groupby
from music21 import pitch
import numpy as np
import matplotlib.pyplot as plt

import util

parser = argparse.ArgumentParser(description='Melody self-similarity')
parser.add_argument('--data', type=str, default='music_data/CMaj_Nottingham/train/',
                    help='location of the data corpus')
parser.add_argument('--melody', type=str, default='jigs_simple_chords_1.mid',
                    help='midi song name')
args = parser.parse_args()


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
    if pitch.Pitch(left[0]) == pitch.Pitch(right[0]):
        diff = 0
    else:
        diff = -1 if pitch.Pitch(left[0]) < pitch.Pitch(right[0]) else 1
    return (diff, right[1])

pdv = util.PitchDurationVocab()
melody, measure_limit = pdv.mid2orig(args.data + args.melody, include_measure_boundaries=False)
melody = melody[1:-1]

differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))
rawDiffs = map(lambda x: x[0], differences)

window = 8

ssm = np.zeros([len(differences), len(differences)])
for i in xrange(window, len(differences)):
    for j in xrange(window, len(differences)):
        ssm[i,j] = edit_distance(rawDiffs[i-window+1:i+1], rawDiffs[j-window+1:j+1])

plt.imshow(ssm, cmap='gray', interpolation='nearest')
plt.show()
plt.savefig('../similarities/' + args.melody + '.png')


