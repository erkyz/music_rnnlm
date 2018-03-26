import argparse
from itertools import groupby
from music21 import pitch
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util
import gen_util

# min window if we use "c"
MIN_WINDOW = 4

pdv = util.PitchDurationVocab()


######################################################################
# Helpers
######################################################################

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


######################################################################
# Not using this stuff right now.
######################################################################

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



######################################################################
# What I'm currently using (other stuff is semi-dead code)
######################################################################


def get_note_ssm_past(melody, args, bnw=False):
    """ self-distance matrix """
    ''' melody is a PDV melody '''
    differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))
    rawDiffs = map(lambda x: x[0], differences)

    ssm = np.zeros([len(differences), len(differences)]) 
    for i in xrange(args.window-1, len(differences)):
        for j in xrange(i, len(differences)):
            if bnw:
                ssm[i,j] = ssm[j,i] = edit_distance(rawDiffs[i-args.window+1:i+1], rawDiffs[j-args.window+1:j+1]) <= args.distance_threshold
            else:
                ssm[i,j] = ssm[j,i] = edit_distance(rawDiffs[i-args.window+1:i+1], rawDiffs[j-args.window+1:j+1]) 

    return ssm, rawDiffs


def get_note_ssm_future(melody, args, bnw=False):
    ''' melody is a PDV melody '''
    differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))
    rawDiffs = map(lambda x: x[0], differences)

    ssm = np.zeros([len(differences), len(differences)]) 
    for i in xrange(0, len(differences)-args.window):
        for j in xrange(i, len(differences)-args.window):
            if bnw:
                ssm[i,j] = ssm[j,i] = edit_distance(rawDiffs[i:i+args.window+1], rawDiffs[j:j+args.window+1]) <= args.distance_threshold
            else:
                # NOTE: this is an SDM
                ssm[i,j] = ssm[j,i] = edit_distance(rawDiffs[i:i+args.window+1], rawDiffs[j:j+args.window+1])

    return ssm, rawDiffs


def get_hid_sim(hiddens, args, bnw=True):
    sims = np.zeros([len(hiddens), len(hiddens)])
    for i in range(len(hiddens)):
        for j in range(i, len(hiddens)):
            l = hiddens[0][i] if args.arch == 'LSTM' else hiddens[i]
            r = hiddens[0][j] if args.arch == 'LSTM' else hiddens[j]
            # cosine similarity
            sims[i,j] = sims[j,i] = (torch.matmul(l,torch.t(r)) / (torch.norm(l) * torch.norm(r) + .000001)).data[0][0] if not torch.equal(l.data, r.data) else 1
    f = np.vectorize(lambda x : x >= 0.95)
    return f(sims) if bnw else sims


def get_rnn_ssm(args, sv, model, events):
    hidden = model.init_hidden(1) 
    gen_data = gen_util.make_data_dict(args, sv)
    hiddens = []
    for t in range(len(events[0])):
        for c in range(sv.num_channels):
            gen_data["data"][c].data.fill_(events[c][t])
        args.epoch = 0
        outputs, hidden = model(gen_data, hidden, args)
        hiddens.append(hidden)
    ssm = get_hid_sim(hiddens, args)
    return ssm


def get_prev_match_idx(ssm, args, sv):
    prev_idxs = []
    # scan left to right. simplified for now to only 0's and 1's, so simpler here too.
    for col in range(ssm.shape[0]):
        row_order = range(col)
        if args.most_recent:
            row_order = reversed(row_order)
        for row in row_order:
            if ssm[row][col] == 1:
                prev_idxs.append(row)
                break
        if len(prev_idxs) == col:
            prev_idxs.append(-1)
        '''
        prev_idxs.append(-1)
        '''
    return prev_idxs


