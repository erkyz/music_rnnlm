import argparse
import sys, os
import shutil
import glob
import ast
import music21
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
parser.add_argument('--num_per_structure_list', type=str, default='[28,8,7]',
        help='number of samples to generate per type of structure_list')
parser.add_argument('--measures_per_section', type=int, default=4)
parser.add_argument('--rm', action='store_true')
args = parser.parse_args()


def get_idxs_where_eq(l, x):
    return [j for j in range(len(l)) if l[j] == x]


# The types of structure_list we want in our synthesized dataset. 
# Each type will have an equal number of examples generated.
structure_lists = ast.literal_eval(args.structure_lists)

out_dir = args.out_dir + '/' + args.genre + args.structure_lists + '/'
if args.rm and os.path.exists(out_dir):
    shutil.rmtree(out_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    os.mkdir(out_dir + 'train')
    os.mkdir(out_dir + 'valid')
    os.mkdir(out_dir + 'test')
else:
    print ("Already done!")
    exit()

num_sections = max([max(l) for l in structure_lists])+1

# For each type of structure_list, we create that structure_list for a song. So each sample is a 
# list of sections.
samples = []


files_read = 0
for i, d in enumerate(['train', 'valid', 'test']):
    num_per_structure_list = ast.literal_eval(args.num_per_structure_list)[i]
    for f in glob.glob(args.data + d + '/' + args.genre + "*.mid"):
        score = music21.converter.parse(f)
        time_signature = util.get_ts(score)
        # TODO what if there aren't clean measure boundaries?? Print and see.
        limit = args.measures_per_section * time_signature.beatCount * \
                    time_signature.beatDuration.quarterLength
        section_num = 0
        section_progress = 0
        section = []
        sample = []
        for part in score:
            for e in part:
                if section_num == num_sections:
                    break
                if section_progress >= limit:
                    sample.append(section)
                    section_num += 1
                    section_progress = 0
                    section = []
                if type(e) is music21.note.Note:
                    section.append((e.nameWithOctave, e.duration.quarterLength))
                    section_progress += e.duration.quarterLength
                elif type(e) is music21.note.Rest:
                    section.append((e.name, e.duration.quarterLength))
                    section_progress += e.duration.quarterLength

        if len(sample) == num_sections:
            info_dict = {
                    'sample': sample,
                    'f': f.split('/')[-1],
                    'ts': time_signature,
                    }
            samples.append(info_dict)
            files_read += 1
        if files_read == num_per_structure_list:
            break

    if files_read < num_per_structure_list:
        print ("-"*88)
        print ("Wanted", num_per_structure_list, "but got", files_read)
        print ("-"*88)

    metas = {}

    for structure_list in structure_lists:
        for sample in samples:
            score = music21.stream.Stream()
            score.append(sample['ts'])

            idx_in_mel = 0
            # Prefix sum, more or less
            sample_starting_idxs = []
            # Create MIDI file
            for i, section_idx in enumerate(structure_list):
                sample_starting_idxs.append(idx_in_mel)
                for tup in sample['sample'][section_idx]:
                    if tup[0] == 'rest':
                        ev = music21.note.Rest()
                    else:
                        ev = music21.note.Note(tup[0])
                    ev.quarterLength = tup[1]
                    score.append(ev)
                    idx_in_mel += 1
            fileNameWithStructure = str(structure_list) + '|' + sample['f']
            newFileName = out_dir + d + '/' + fileNameWithStructure
            score.write('midi', newFileName)
            sample_len = idx_in_mel

            # Get SSM
            ssm = np.zeros([sample_len, sample_len])
            idx_in_mel = 0
            for i in range(len(structure_list)):
                section_idx = structure_list[i]
                idxs_eq = get_idxs_where_eq(structure_list, section_idx)
                for j in range(len(sample['sample'][section_idx])):
                    for eq_idx in idxs_eq:
                        ssm[idx_in_mel, sample_starting_idxs[eq_idx] + j] = 1
                    idx_in_mel += 1
            metas[fileNameWithStructure] = {'ssm': ssm}

    with open(out_dir + '/' + d + '/meta.p', 'wb') as f:
        pickle.dump(metas, f, protocol=2)


