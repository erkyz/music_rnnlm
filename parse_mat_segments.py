# Parse the matlab-miditoolbox-created segments in a dataset
# Save to new meta.p file

import sys
import pickle
sys.path.insert(0, 'torch_models')
import similarity, util
import music21

d = 'music_data/CMaj_Nottingham/train/'

metas = {}
with open(d + 'segments.txt') as segs:
    for l in segs:
        if l[0] != '1':
            f = l[:-1]

        print (d+f)
        skip_song = False
        out = []
        score = music21.converter.parse(d + f)
        time_signature = util.get_ts(score)
        limit = time_signature.beatCount * time_signature.beatDuration.quarterLength
        section_progress = 0
        note_num = 0
        measure_starting_idxs = [0]

        pickup_dur = 0
        pickup_len = 0
        if len(score) == 1:
            print "Skip song, no chords"
            continue
        if type(score[1][2]) is music21.note.Rest:
            pickup_dur = score[1][2].duration.quarterLength
         
        on_pickup = True
        time_signature_encountered = False
        for e in score[0]: # this is the melody Part
            # Effectively throw out the pickup.
            if on_pickup and type(e) in {music21.note.Note, music21.note.Rest}:
                pickup_len += 1
                if section_progress < pickup_dur:
                    section_progress += e.duration.quarterLength
                    continue
                elif section_progress == pickup_dur:
                    section_progress = 0
                    on_pickup = False
        
        for e in score[0]: # this is the melody Part
            if section_progress > limit:
                assert(False)
            if section_progress == limit:
                section_progress = 0
            if type(e) is music21.meter.TimeSignature:
                if time_signature_encountered:
                    print "Skip song"
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
                    skip_song = True
                    print "Skip song"
                    break
                else:
                    # add_to_section(section, name, duration)
                    section_progress += duration 
                note_num += 1

        if not skip_song:
            # Offset |starts| by the number of pickup notes we threw out.
            starts = [int(x)-pickup_len-1 for x in l.split(' ')[:-1]]
            if len(starts) == 0:
                continue
            starts[0] = 0
            ends = starts[1:] + [-1] # last end is -1
            segments = list(zip(starts, ends))
            print segments
            segment_sdm = similarity.get_measure_sdm(out, segments)
            print segment_sdm
            metas[f] = {'segments': list(zip(starts, ends)), 'f': f, 'segment_sdm': segment_sdm}

pickle.dump(metas, open(d + 'segment_meta.p', 'wb'))
print len(metas.keys())

