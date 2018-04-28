import argparse
import sys, os
import shutil
import glob
import ast
import music21
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import itertools
import pickle
import random

import util

# USE PYTHON3 FOR THIS BECAUSE MUSIC21 FOR PYTHON 2 IS OUTDATED.
if sys.version_info[0] != 3:
    print("Use Python3!")
    exit()

parser = argparse.ArgumentParser(description='Synthesize musical examples')

# TODO
parser.add_argument('--data', type=str, default='music_data/CMaj_Nottingham/',
                    help='location of the data corpus to sample from')
parser.add_argument('--out_dir', type=str, default='music_data')
parser.add_argument('--genre', type=str, default='ashover',
           help='(Nottingham-only for now): prefix of song names to sample from.')
parser.add_argument('--structure_lists', type=str,
                    help='list of (ABA structure_list in a list of 0-indexed ints)')
parser.add_argument('--num_per_structure_list', type=str, default='[100,10,10]',
        help='number of samples to generate per type of structure_list')
parser.add_argument('--measures_per_section', type=int, default=4)
parser.add_argument('--rm', action='store_true')
args = parser.parse_args()


def get_idxs_where_eq(l, x):
    return [j for j in range(len(l)) if l[j] == x]

def add_to_section(section, name, duration):
    if len(section) > 0 and section[-1][0] == 'rest' and name == 'rest':
        prev_rest_dur = section[-1][1]
        section = section[:-1]
        section.append((name, duration+prev_rest_dur))
    else:
        section.append((name, duration))
    return section 

# The types of structure_list we want in our synthesized dataset. 
# Each type will have an equal number of examples generated.
structure_lists = ast.literal_eval(args.structure_lists)

out_dir = args.out_dir + '/' + args.genre + args.structure_lists + '/'
if args.rm and os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
os.mkdir(out_dir + 'train')
os.mkdir(out_dir + 'valid')
os.mkdir(out_dir + 'test')

MIN_SECTION_LEN = 4
num_per_structure_list = ast.literal_eval(args.num_per_structure_list)
num_sections = max([max(l) for l in structure_lists])+1
ts_counter = Counter()
ts_sections = {}

for i, d in enumerate(['train', 'valid', 'test']):
    # For each type of structure_list, we create that structure_list for a song. So 
    # each sample is a list of sections.

    for f in glob.glob(args.data + d + '/' + args.genre + "*.mid"):
        score = music21.converter.parse(f)
        time_signature = util.get_ts(score)
        limit = time_signature.beatCount * time_signature.beatDuration.quarterLength
        section_progress = 0
        section = []
        for part in score:
            for e in part:
                if section_progress > limit:
                    assert(False)
                if section_progress == limit:
                    if len(section) >= MIN_SECTION_LEN:
                        ts_sections.setdefault(time_signature, []).append(section)
                        ts_counter[time_signature] += 1
                    section_progress = 0
                    section = []
                if type(e) is music21.note.Note or type(e) is music21.note.Rest:
                    duration = e.duration.quarterLength
                    name = e.nameWithOctave if type(e) is music21.note.Note else 'rest'
                    if section_progress + duration > limit:
                        cut_duration = limit - section_progress
                        add_to_section(section, name, cut_duration)
                        ts_counter[time_signature] += 1
                        ts_sections.setdefault(time_signature, []).append(section)
                        
                        section = []
                        section_progress = section_progress + duration - limit
                        add_to_section(section, name, section_progress)
                    else:
                        add_to_section(section, name, duration)
                        section_progress += duration 

# get most frequent time signature
print(ts_counter)
ts, _ = ts_counter.most_common(1)[0]

for subdir_i, d in enumerate(['train', 'valid', 'test']):
    metas = {}
    for structure_list in structure_lists:
        for num_generated in range(num_per_structure_list[subdir_i]):
            '''
            # pick one of the time signatures
            i = random.randrange(sum(ts_counter.values()))
            ts = next(itertools.islice(ts_counter.elements(), i, None))
            '''
            sample = []
            # sample len(structure_list) sections for this time signature
            for idx in range(len(structure_list)):
                next_section = random.choice(ts_sections[ts])
                while idx > 0 and next_section in sample:
                    next_section = random.choice(ts_sections[ts])
                sample.append(next_section)

            info_dict = {
                    'sample': sample,
                    'ts': time_signature,
                    }

            score = music21.stream.Stream()
            score.append(ts)

            idx_in_mel = 0
            sample_starting_idxs = []
            sample_ending_idxs = []
           
            # Create MIDI and intermeidate events 
            evs = []
            origs = []
            for i, section_idx in enumerate(structure_list):
                sample_starting_idxs.append(idx_in_mel)
                sample_ending_idxs.append(idx_in_mel)
                for tup in sample[section_idx]:
                    if tup[0] == 'rest':
                        ev = music21.note.Rest()
                    else:
                        ev = music21.note.Note(tup[0])
                    ev.quarterLength = tup[1]
                    origs.append(tup)
                    score.append(ev)
                    evs.append(ev)
                    idx_in_mel += 1
            # TODO rename this variable
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
                section_idx = structure_list[i]
                idxs_eq = get_idxs_where_eq(structure_list, section_idx)
                for j in range(len(sample[section_idx])):
                    for eq_idx in idxs_eq:
                        ssm[idx_in_mel, sample_starting_idxs[eq_idx] + j] = 1
                    idx_in_mel += 1

            # Account for START and END
            x = ssm.shape[0]
            ssm = np.insert(ssm, 0, 0, axis=0)
            ssm = np.insert(ssm, 0, 0, axis=1)
            ssm[0,0] = 1
            ssm[-1,-1] = 1

            indices_of_sections = [[j for j in range(len(structure_list)) if structure_list[j] == i] for i in range(num_sections)]
            repeating_sections = [x for x in indices_of_sections if len(x) > 1]
            
            avg_ed = 5

            # Create a len(structure_lis)^2 SSM (1 or 0)
            segment_sdm = [[0 if structure_list[j] == structure_list[i] else avg_ed for j in range(len(structure_list))] for i in range(len(structure_list))]

            # Note that segments doesn't include START or END.
            metas[fileNameWithStructure] = {
                    'origs': origs,
                    'segments': list(zip(sample_starting_idxs, sample_ending_idxs)),
                    'ssm': ssm, 
                    'segment_sdm': segment_sdm, 
                    'repeating_sections': repeating_sections,
                    'ts': ts,
                    }

    with open(out_dir + '/' + d + '/meta.p', 'wb') as f:
        pickle.dump(metas, f, protocol=2)


