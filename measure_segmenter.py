# For Nottingham.
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
sys.path.insert(0, 'torch_models')
import util, similarity

parser = argparse.ArgumentParser(description='Synthesize musical examples')
parser.add_argument('--data', type=str, default='music_data/CMaj_Nottingham/',
                    help='location of the data corpus to sample from')
args = parser.parse_args()

ties = []
multiple_ts = []
no_chords = []
for i, d in enumerate(['test', 'valid', 'train']):
    metas = {}
    for f in glob.glob(args.data + d + '/' + "*.mid"):
        print f
        out = []
        score = music21.converter.parse(f)
        time_signature = util.get_ts(score)
        limit = time_signature.beatCount * time_signature.beatDuration.quarterLength
        section_progress = 0
        note_num = 0
        measure_starting_idxs = [0]

        # NOTE this is VERY Nottingham-specific :(
        # First, get the chord part. This determines how long the pickup is. 
        # We're going to ignore the pickup in each song for clean measures.

        pickup_dur = 0
        if len(score) == 1:
            no_chords.append(f)
            continue
        if type(score[1][2]) is music21.note.Rest:
            pickup_dur = score[1][2].duration.quarterLength
        print pickup_dur
         
        on_pickup = True
        time_signature_encountered = False
        for e in score[0]: # this is the melody Part
            # Throw out the pickup.
            if on_pickup and type(e) in {music21.note.Note, music21.note.Rest}:
                if section_progress < pickup_dur:
                    section_progress += e.duration.quarterLength
                    continue
                elif section_progress == pickup_dur:
                    section_progress = 0
                    on_pickup = False

            if section_progress > limit:
                assert(False)
            if section_progress == limit:
                section_progress = 0
                measure_starting_idxs.append(note_num)
            if type(e) is music21.meter.TimeSignature:
                if time_signature_encountered:
                    multiple_ts.append(f)
                    break
                time_signature_encountered = True
            if type(e) is music21.note.Note or type(e) is music21.note.Rest:
                duration = e.duration.quarterLength
                name = e.nameWithOctave if type(e) is music21.note.Note else 'rest'
                if type(e) is music21.note.Note:
                    out.append((e.nameWithOctave, e.duration.quarterLength))
                elif type(e) is music21.note.Rest:
                    out.append((e.name, e.duration.quarterLength))
                if section_progress + duration > limit:
                    ties.append(f)
                    break
                    '''
                    cut_duration = limit - section_progress
                    # add_to_section(section, name, cut_duration)
                    ts_sections.setdefault(time_signature, []).append(section)
                    
                    section_progress = section_progress + duration - limit
                    # add_to_section(section, name, section_progress)
                    '''
                else:
                    # add_to_section(section, name, duration)
                    section_progress += duration 
                note_num += 1
        print measure_starting_idxs
        ends = measure_starting_idxs[1:] + [note_num+1]
        basename = os.path.basename(f) 
        segments = list(zip(measure_starting_idxs, ends))
        segment_sdm = similarity.get_measure_sdm(out, segments)
        metas[basename] = {'segments': segments, 'f': basename, 'segment_sdm': segment_sdm}

    pickle.dump(metas, open(args.data + d + '/meta.p', 'wb'))

print ties
print len(ties)
print multiple_ts
print len(multiple_ts)
print no_chords
print len(no_chords)


