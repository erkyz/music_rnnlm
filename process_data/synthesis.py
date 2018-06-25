import sys, os
import argparse
import shutil
import glob
import ast
import music21
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import random
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util, similarity

# NOTE: MUST USE PYTHON3 WHEN RUNNING THIS SCRIPT BECAUSE MUSIC21 FOR PYTHON2 IS DEPRECATED.
# Run "pip3 install music21"
if sys.version_info[0] != 3:
    print("Use Python3!")
    exit()

parser = argparse.ArgumentParser(description='Synthesize musical examples')

parser.add_argument('--data', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus to sample from')
parser.add_argument('--out_dir', type=str, default='../music_data/synthesized')
parser.add_argument('--genre', type=str, default='',
        help='prefix of song names to sample from.')
parser.add_argument('--structure_lists', type=str,
        help='list of (ABA structure_list in a list of 0-indexed ints), provided as a string')
parser.add_argument('--num_per_structure_list', type=str, default='[100,10,10]',
        help='number of samples [train,valid,test] to generate per type of structure_list')
parser.add_argument('--min_measure_len', type=int, default=4,
        help="")
parser.add_argument('--num_ts_to_use', type=int, default=50, 
        help="Use the X most common time signatures in the corpus for use in synthesis.")
parser.add_argument('--rm', action='store_true', 
        help="Whether to remove the existing data in args.out_dir")
args = parser.parse_args()


def get_idxs_where_eq(l, x):
    return [j for j in range(len(l)) if l[j] == x]

def add_to_measure(measure, name, duration):
    if len(measure) > 0 and measure[-1][0] == 'rest' and name == 'rest':
        prev_rest_dur = measure[-1][1]
        measure = measure[:-1]
        measure.append((name, duration+prev_rest_dur))
    else:
        measure.append((name, duration))
    return measure 

# The types of structure_list we want in our synthesized dataset. 
# Each type will have an equal number of examples generated.
structure_lists = ast.literal_eval(args.structure_lists)

out_dir = args.out_dir
if args.rm and os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
os.mkdir(out_dir + 'train')
os.mkdir(out_dir + 'valid')
os.mkdir(out_dir + 'test')

num_per_structure_list = ast.literal_eval(args.num_per_structure_list)
num_measures = max([max(l) for l in structure_lists])+1
if os.path.exists(args.data + 'measures.p'):
    ts_measures, ts_counter = pickle.load(open(args.data + 'measures.p', 'rb'))
else:
    ts_counter = Counter()
    ts_measures = {}

    for i, d in enumerate(['train', 'valid', 'test']):
        # For each type of structure_list, we create that structure_list for a song. So 
        # each sample is a list of measures.

        for f in glob.glob(args.data + d + '/' + "*.mid"):
            score = music21.converter.parse(f)
            time_signature = util.get_ts(score)
            limit = time_signature.beatCount * time_signature.beatDuration.quarterLength
            measure_progress = 0
            measure = []

            # Note: Some files have a pickup that throws off measure-segmentation.
            pickup_dur = 0
            if len(score) == 1:
                continue
            if type(score[1][2]) is music21.note.Rest:
                pickup_dur = score[1][2].duration.quarterLength
            # print (pickup_dur)

            on_pickup = True
            time_signature_encountered = False
            for e in score[0]: # Note: We assume this is the melody Part
                # Throw out the pickup.
                if on_pickup and type(e) in {music21.note.Note, music21.note.Rest}:
                    if measure_progress < pickup_dur:
                        measure_progress += e.duration.quarterLength
                        continue
                    elif measure_progress == pickup_dur:
                        measure_progress = 0
                        on_pickup = False

                if measure_progress > limit:
                    assert(False)
                if measure_progress == limit:
                    if len(measure) >= args.min_measure_len:
                        ts_measures.setdefault(time_signature, []).append(measure)
                        ts_counter[time_signature] += 1
                    measure_progress = 0
                    measure = []
                if type(e) is music21.meter.TimeSignature:
                    if time_signature_encountered:
                        break
                    time_signature_encountered = True
                if type(e) is music21.note.Note or type(e) is music21.note.Rest:
                    duration = e.duration.quarterLength
                    name = e.nameWithOctave if type(e) is music21.note.Note else 'rest'
                    if measure_progress + duration > limit:
                        break
                    else:
                        add_to_measure(measure, name, duration)
                        measure_progress += duration 
    
    pickle.dump((ts_measures, ts_counter), open(args.data + 'measures.p', 'wb'))

# Get most frequent time signatures
print(ts_counter)

for subdir_i, d in enumerate(['train', 'valid', 'test']):
    metas = {}
    for structure_list in structure_lists:
        for num_generated in range(num_per_structure_list[subdir_i]):
            ts, _ = ts_counter.most_common(args.num_ts_to_use)[num_generated % args.num_ts_to_use]
            sample = []
            # sample len(structure_list) measures for this time signature
            for idx in range(len(structure_list)):
                next_measure = random.choice(ts_measures[ts])
                while idx > 0 and next_measure in sample:
                    next_measure = random.choice(ts_measures[ts])
                sample.append(next_measure)

            info_dict = {
                    'sample': sample,
                    'ts': ts,
                    }

            score = music21.stream.Stream()
            score.append(ts)

            idx_in_mel = 0
            sample_starting_idxs = []
            sample_ending_idxs = []
           
            # Create MIDI and intermeidate events 
            evs = []
            origs = []
            for i, measure_idx in enumerate(structure_list):
                sample_starting_idxs.append(idx_in_mel)
                sample_ending_idxs.append(idx_in_mel)
                for tup in sample[measure_idx]:
                    if tup[0] == 'rest':
                        ev = music21.note.Rest()
                    else:
                        ev = music21.note.Note(tup[0])
                    ev.quarterLength = tup[1]
                    origs.append(tup)
                    score.append(ev)
                    evs.append(ev)
                    idx_in_mel += 1
            fileNameWithStructure = str(structure_list) + str(num_generated) + '.mid'
            newFileName = out_dir + d + '/' + fileNameWithStructure
            score.write('midi', newFileName)
            sample_len = idx_in_mel
            sample_ending_idxs = sample_ending_idxs[1:]
            sample_ending_idxs.append(idx_in_mel)

            # Get SSM
            ssm = np.zeros([sample_len+1, sample_len+1])
            idx_in_mel = 0
            for i in range(len(structure_list)):
                measure_idx = structure_list[i]
                idxs_eq = get_idxs_where_eq(structure_list, measure_idx)
                for j in range(len(sample[measure_idx])):
                    for eq_idx in idxs_eq:
                        ssm[idx_in_mel, sample_starting_idxs[eq_idx] + j] = 1
                    idx_in_mel += 1

            # Account for START and END tokens
            x = ssm.shape[0]
            ssm = np.insert(ssm, 0, 0, axis=0)
            ssm = np.insert(ssm, 0, 0, axis=1)
            ssm[0,0] = 1
            ssm[-1,-1] = 1

            indices_of_measures = [[j for j in range(len(structure_list)) if structure_list[j] == i] for i in range(num_measures)]
            repeating_measures = [x for x in indices_of_measures if len(x) > 1]

            measure_boundaries = list(zip(sample_starting_idxs, sample_ending_idxs))
            # Note that |measure_sdm| is currently unused in models because we learn a scoring function instead.
            measure_sdm = similarity.get_measure_sdm(origs, measure_boundaries)

            # Note that measures doesn't include START or END token, but these are accounted for in the SSM. (Rife for OBO errors, sadly)
            metas[fileNameWithStructure] = {
                    'origs': origs,                           # original data
                    'measure_boundaries': measure_boundaries, # start+end idx of each measure
                    'ssm': ssm,                               # SSM for the song
                    'measure_sdm': measure_sdm,               # measure-level SDM
                    'repeating_measures': repeating_measures, # which measures repeat (eg [[0,2]])
                    'ts': ts,                                 # this song's TS (in music21)
                    }

    with open(out_dir + '/' + d + '/meta.p', 'wb') as f:
        # Dump so it can be read using Python 2
        pickle.dump(metas, f, protocol=2)

print ("Done!")


