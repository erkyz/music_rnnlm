import sys, os
import argparse
import glob
import ast
import music21
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import random
import signal
sys.path.insert(0, 'torch_models')
import util, similarity

parser = argparse.ArgumentParser(description='Measure parsing')
parser.add_argument('--data', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus to sample from')
parser.add_argument('--out', type=str, default='../music_data/guitar_melodies/',
                    help='where to put melodies')
parser.add_argument('--max_note_num', type=int, default=450)

args = parser.parse_args()

num_negative = 0

def handler(signum, frame):
    pass

def add_prev_note(out, prev_note, curr_t, measure_starting_idxs, note_num, limit):
    add_rest = None
    measure_num = len(measure_starting_idxs)
    bar_location = measure_num*limit
    if prev_note.offset != 0 and curr_t%limit == 0:
        print ("MEASURE", prev_note.offset, limit)
        measure_starting_idxs.append(note_num)
        duration = curr_t-prev_note.offset
    elif prev_note.offset != 0 and int(curr_t/limit) > int(prev_note.offset/limit):
        print ("TIE", curr_t, prev_note.offset, limit)
        # Make the tie instead end at the end of the measure.
        duration = bar_location - prev_note.offset + 1
        add_rest = curr_t - bar_location - 1
        measure_starting_idxs.append(note_num+1)
        if duration < 0:
            return None, None, None # Shitty exception.
    else:
        duration = curr_t-prev_note.offset

    name = prev_note.nameWithOctave if type(prev_note) is music21.note.Note else 'rest'
    if type(prev_note) is music21.chord.Chord:
        out.append((prev_note.pitches[-1].nameWithOctave, duration))
    elif type(prev_note) is music21.note.Note:
        out.append((prev_note.nameWithOctave, duration))
    elif type(prev_note) is music21.note.Rest:
        out.append((prev_note.name, duration%limit))
    note_num += 1

    if add_rest is not None:
        out.append(('rest', add_rest))
        note_num += 1
     
    return out, measure_starting_idxs, note_num


def get_next_ts(part):
    for e in part: # this is the melody Part
        if type(e) is music21.meter.TimeSignature:
            try:
                return e, e.beatCount * e.beatDuration.quarterLength
            except Exception:
                print ("TimeSignatureException Again")
                return None, None

def process(part, note_num, out, limit, skip_song, pickup_dur, ts):
    global num_negative

    curr_t = 0
    measure_starting_idxs = [0]
    prev_note = None
    dur_since_prev = 0
    print ("limit is", limit)

    for i_e, e in enumerate(part): # this is the melody Part
        # Throw out the pickup for Nottingham.
        # Commented out if we're not working with non-Nottingham music.
        # TODO add a flag.
        '''
        if on_pickup and type(e) in {music21.note.Note, music21.note.Rest, music21.chord.Chord}:
            if section_progress < pickup_dur:
                section_progress += e.duration.quarterLength
                continue
            elif section_progress == pickup_dur:
                section_progress = 0
                on_pickup = False
        '''
        if type(e) is music21.meter.TimeSignature:
            if limit is not None: 
                # This is not the first TS
                print ("Multiple TS")
                multiple_ts.append(f)
                skip_song = True
                break
            try:
                limit = e.beatCount * e.beatDuration.quarterLength 
                ts = e
            except Exception:
                pickup_dur = e.beatCount
                print ("TimeSignatureException")
                ts, limit = get_next_ts(part[i_e+1:])

        if type(e) is music21.stream.Voice:
            return process(e, note_num, out, limit, skip_song, pickup_dur, ts)

        if e.offset < pickup_dur:
            print ("skip")
            continue
        else:
            e.offset -= pickup_dur
            print (e, e.offset, note_num)

        if limit is not None: 
            # We found a TS.
            if type(e) in {music21.note.Note, music21.chord.Chord, music21.note.Rest}:
                # Add prev note
                curr_t = e.offset
                if prev_note is not None:
                    out, measure_starting_idxs, note_num = add_prev_note(
                            out, prev_note, curr_t,
                            measure_starting_idxs, note_num, limit)
                    if out is None:
                        # We should skip here.
                        num_negative += 1
                        print ("NEGATIVE DURATION", num_negative)
                        return [], [], True, 0, None
                prev_note = e
            if note_num == args.max_note_num:
                break
    if len(out) > args.max_note_num:
        assert(False)
    return out, measure_starting_idxs, skip_song, note_num, ts


multiple_ts = []
no_chords = []
num_skipped = 0
for i, d in enumerate(['test', 'valid', 'train']):
    metas = {}
    for f in glob.glob(args.data + d + '/' + "*.mid"):
        print ("Segment", f)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(30)
        try:
            # Sometimes this function hangs.
            score = music21.converter.parse(f) 
        except Exception:
            # I don't think solution works, though.
            print ("Took too long") 
        out = []

        limit = None
        skip_song = False
        ts = None
        pickup_dur = 0
        note_num = 1
        out, measure_starting_idxs, skip_song, note_num, ts = process(
                score[0], note_num, out, limit, skip_song, pickup_dur, ts)
        skip_song = skip_song or ts is None
        
        if not skip_song:
            print ("Segmented", f)
            print (measure_starting_idxs)
            ends = measure_starting_idxs[1:] + [note_num+1]
            basename = os.path.basename(f) 
            measure_boundaries = list(zip(measure_starting_idxs, ends))
            measure_sdm = similarity.get_measure_sdm(out, measure_boundaries)
            metas[basename] = {
                    'measure_boundaries': measure_boundaries, 
                    'f': basename, 
                    'measure_sdm': measure_sdm
            }

            s = music21.stream.Stream()
            s.append(ts) 
            print (out)
            for e in out:
                if e[0] == 'rest':
                    n = music21.note.Rest()
                else:
                    n = music21.note.Note(e[0])
                n.quarterLength = e[1]
                s.append(n)
            mf = music21.midi.translate.streamToMidiFile(s)
            mf.open(args.out + d + '/' + basename, 'wb')
            mf.write()
            mf.close()
        else:
            num_skipped += 1
            print ("NUM SKIPPED:", num_skipped)

    pickle.dump(metas, open(args.out + d + '/meta.p', 'wb'))

print (multiple_ts)
print (len(multiple_ts))
print (no_chords)
print (len(no_chords))

