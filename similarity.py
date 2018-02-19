import argparse
from itertools import groupby
from music21 import pitch
import numpy as np
import matplotlib.pyplot as plt

import util

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__
    return helper
def memoize(func):
    mem = {}
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]
    return memoizer
@call_counter
@memoize   
def edit_distance(src, dst):
    if not dst: return len(src)
    if not src: return len(dst)

    return min(
            # Case 1
            (edit_distance(src[:-1], dst[:-1]) + (1, 0)[src[-1] == dst[-1]]),
            # Case 2
            (edit_distance(src[:-1], dst) + 1),
            # Case 3
            (edit_distance(src, dst[:-1]) + 1)
    ) 


# padding = 0
# right before measure = 1
# right after measure = 2
# right before rest = 3
# right after rest = 4
# stay = 5
# up = 6
# down = 7

def diff(x):
    left, right = x
    if left[0] == 'rest': return (3, right[1])
    if right[0] == 'rest': return (4, right[1])
    if left[0] == 'measure': return (2, right[1])
    if right[0] == 'measure': return (1, right[1])
    if left[0] == 'padding' or right[0] == 'padding' or left[0] == 'end' or right[0] == 'end': 
        return (0, right[1])
    if pitch.Pitch(left[0]) == pitch.Pitch(right[0]):
        return (5, right[1])
    else:
        diff = 6 if pitch.Pitch(left[0]) < pitch.Pitch(right[0]) else 7
    return (diff, right[1])

def get_avg_dist_between_measures(melody, sv):
    measure_counts = []
    c = 0
    for m, _ in melody:
        if m == 'measure':
            measure_counts += [c]
            c = 0
        else:
            c += 1
    return sum(measure_counts)/len(measure_counts)

'''
def diff(x):
    left, right = x
    if left[0] == 'rest' or right[0] == 'rest':
        # TODO temporary
        return (0, right[1])
    # TODO LOL this is awful
    if left[0] == 'padding' or right[0] == 'padding' or left[0] == 'measure' or right[0] == 'measure' or left[0] == 'end' or right[0] == 'end': 
        return (0, right[1])
    if pitch.Pitch(left[0]) == pitch.Pitch(right[0]):
        diff = 0
    else:
        diff = -1 if pitch.Pitch(left[0]) < pitch.Pitch(right[0]) else 1
    return (diff, right[1])
'''

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

LARGE_DISTANCE = 20

def get_note_sdm(melody, window):
    """ self-distance matrix """
    ''' melody is a PDV melody '''
    differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))
    rawDiffs = map(lambda x: x[0], differences)

    sdm = np.ones([len(differences), len(differences)]) * LARGE_DISTANCE
    for i in xrange(window-1, len(differences)):
        for j in xrange(i, len(differences)):
            sdm[i,j] = sdm[j,i] = edit_distance(rawDiffs[i-window:i], rawDiffs[j-window:j])

    return sdm, rawDiffs

def get_note_ssm(melody, args):
    """ self-distance matrix """
    ''' melody is a PDV melody '''
    differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))
    rawDiffs = map(lambda x: x[0], differences)

    ssm = np.zeros([len(differences), len(differences)]) 
    for i in xrange(args.window-1, len(differences)):
        for j in xrange(i, len(differences)):
            # TODO this is maybe temporary, just do 1's and 0's
            ssm[i,j] = ssm[j,i] = 1 if edit_distance(rawDiffs[i-args.window+1:i+1], rawDiffs[j-args.window+1:j+1]) <= args.distance_threshold else 0

    return ssm, rawDiffs

MIN_WINDOW = 6

def get_prev_match_idx(melody, args, sv):
    args.window = max(args.c*int(get_avg_dist_between_measures(melody, sv)), MIN_WINDOW)
    ssm, _ = get_note_ssm(melody, args)
    prev_idxs = []
    # scan left to right. simplified for now to only 0's and 1's, so simpler here too.
    for col in range(ssm.shape[0]-1):
        row_order = range(0,col)
        if args.most_recent:
            row_order = reversed(row_order)
        for row in row_order:
            if ssm[row][col] == 1:
                prev_idxs.append(row)
                break
        if len(prev_idxs) == col:
            prev_idxs.append(-1)
    return prev_idxs


def get_min_past_distance(melody, args):
    ''' get the index for the event in the past that's most similar to the current event, '''
    ''' for every event '''
    sdm, _ = get_note_sdm(melody, args.window)
    prev_idxs = []
    # for each column, get the minimum before i
    for row in range(sdm.shape[0]-1):
        if sdm[row][:row].size > 0 and np.amin(sdm[row][:row]) < args.distance_threshold:
            prev_idxs.append(np.argmin(sdm[row][:row]))
        else:
            # provide no information
            prev_idxs.append(-1)
    return prev_idxs


def get_future_from_past(melody, args):
    ''' get the prediction for the next value based on the most similar sequence in the past '''
    ''' melody is a PDV melody ''' # TODO don't do that. lol.
    sdm, diffs = get_note_sdm(melody, args.window)
    # for each column, get the minimum before i
    future_preds = []
    for row in range(sdm.shape[0]-1):
        if sdm[row][:row].size > 0 and np.amin(sdm[row][:row]) < args.distance_threshold:
            prev_idx = np.argmin(sdm[row][:row])
            # get whether the next note is predicted to be up, down, or the same.
            differential = diffs[prev_idx+1] # in {-1,0,1}
            future_preds.append(differential)
        else:
            # otherwise, provide no information
            future_preds.append(2)
    return [x+1 for x in future_preds] # in {0,1,2,3}



