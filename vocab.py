import pickle, os
import ast
import music21
from collections import defaultdict

import events
import util
from constants import * 

class SimpleVocab(object):
    '''
    - Simple parent class for all vocabularies.
    - Designed for multiple "channels," which allows you to factorize an event into
        channels, for instance, pitch and duration, each with their own vocab.
    '''
    
    def __init__(self, num_channels=1):
        self.events = [set([]) for i in range(num_channels)]
        self.origs = [set([]) for i in range(num_channels)]
        self.i2e = [defaultdict(events.Event.not_found) for i in range(num_channels)]
        self.orig2e = [defaultdict(events.Event.not_found) for i in range(num_channels)]
        self.special_events = {}

    @property
    def num_channels(self):
        return len(self.events)
      
    @property
    def sizes(self):
        ''' Returns the size of each of the channel vocabularies '''
        return [len(self.events[c]) for c in range(self.num_channels)]

    def add_event_to_all(self, orig):
        ''' Adds an event (the original event, not an index) to all channels '''
        if orig in self.origs[0]: # TODO
            return self.orig2e[0][orig]
        for channel_idx in range(self.num_channels):
            event_idx = self.sizes[channel_idx]
            e = events.Event(event_idx, orig)
            self.i2e[channel_idx][event_idx] = e
            self.orig2e[channel_idx][orig] = e
            self.events[channel_idx].add(e)
            self.origs[channel_idx].add(orig)
        return e

    def add_event_to_channel(self, orig, channel_idx):
        if orig in self.origs[channel_idx]:
            return self.orig2e[channel_idx][orig]
        event_idx = self.sizes[channel_idx]
        e = events.Event(event_idx, orig)
        self.i2e[channel_idx][event_idx] = e
        self.orig2e[channel_idx][orig] = e
        self.events[channel_idx].add(e)
        self.origs[channel_idx].add(orig)
        return e

    def __getitem__(self, key, channel=0):
        if isinstance(key, int): return self.i2e[channel][key]
        elif isinstance(key, events.Event): return key
        else: return self.orig2e[channel][key]

    def save(self, filename):
        info_dict = {
                "events": self.events,
                "origs": self.origs,
                "i2e": [dict(self.i2e[c]) for c in range(self.num_channels)],
                "orig2e": [dict(self.orig2e[c]) for c in range(self.num_channels)],
                "special_events": self.special_events,
                }
        with open(filename, "w") as f: pickle.dump(info_dict, f)

    @classmethod
    def load(clss, filename):
        with open(filename, "r") as f:
            info_dict = pickle.load(f)
            v = clss()
            v.events = info_dict["events"]
            v.origs = info_dict["origs"]
            v.i2e = [defaultdict(events.Event.not_found, info_dict["i2e"][i]) \
                        for i in range(v.num_channels)]
            v.orig2e = [defaultdict(events.Event.not_found, info_dict["orig2e"][i]) \
                            for i in range(v.num_channels)]
            v.special_events = info_dict["special_events"]
            print ("Vocab sizes:", v.sizes)
            return v


class PitchDurationVocab(SimpleVocab):
    def __init__(self):
        super(PitchDurationVocab, self).__init__(num_channels=1)
        self.special_events = {
                "padding": self.add_event_to_all((PADDING_NAME, PADDING_NAME)),
                "start": self.add_event_to_all((START_OF_TRACK_NAME, START_OF_TRACK_NAME)),
                "end": self.add_event_to_all((END_OF_TRACK_NAME, END_OF_TRACK_NAME)),
                "measure": self.add_event_to_all((MEASURE_NAME, MEASURE_NAME)),
                }

    @classmethod
    def mid2orig(clss, midf, include_measure_boundaries, channel):
        score = music21.converter.parse(midf)
        out = [(START_OF_TRACK_NAME, START_OF_TRACK_NAME)]
        time_signature = util.get_ts(score)
        for part in score:
            for e in part:
                if type(e) is music21.note.Note:
                    out.append((e.nameWithOctave, e.duration.quarterLength))
                elif type(e) is music21.note.Rest:
                    out.append((e.name, e.duration.quarterLength))
            break # TODO
        out.append((END_OF_TRACK_NAME, END_OF_TRACK_NAME))
        return out, 0

    @classmethod
    def load_from_pickle(clss, path, vocab_fname):
        if os.path.isfile(vocab_fname):
            return clss.load(vocab_fname)
        v = clss()
        # note that measure token is already included
        for d in ['train', 'valid', 'test']:
            for _, meta_dict in pickle.load(open(path + d + '/meta.p', 'rt')).iteritems():
                events = meta_dict['origs']
                for name, duration in events:
                    v.add_event_to_all((str(name), duration))
        print ("PitchDurationVocab sizes:", v.sizes)
        v.save(vocab_fname)
        return v


    @classmethod
    def load_from_corpus(clss, paths, vocab_fname):
        if os.path.isfile(vocab_fname):
            return clss.load(vocab_fname)
        v = clss()
        for path in ast.literal_eval(paths):

            filenames = util.getmidfiles(path) 
            for filename in filenames:
                # note that measure token is already included
                events, _ = clss.mid2orig(filename, include_measure_boundaries=False, channel=0)
                for event in events:
                    v.add_event_to_all(event)
        print ("PitchDurationVocab sizes:", v.sizes)
        v.save(vocab_fname)
        return v

    def events2mid(self, l, out):
        # TODO this is a little strange because l will be a list of lists in our API
        # to allow other classes to have multiple channels
        s = music21.stream.Stream()
        for e in l[0]:
            if e == self.special_events["end"]:
                break
            if e in self.special_events.values():
                continue
            if e.original[0] == 'rest':
                n = music21.note.Rest()
            else:
                n = music21.note.Note(e.original[0])
            n.quarterLength = e.original[1]
            s.append(n)
        mf = music21.midi.translate.streamToMidiFile(s)
        mf.open(out, 'wb')
        mf.write()
        mf.close()


class FactorPitchDurationVocab(SimpleVocab):
    def __init__(self):
        super(FactorPitchDurationVocab, self).__init__(num_channels=2) 
        self.special_events = {
                "padding": self.add_event_to_all(PADDING_NAME),
                "start": self.add_event_to_all(START_OF_TRACK_NAME),
                "end": self.add_event_to_all(END_OF_TRACK_NAME),
                "measure": self.add_event_to_all(MEASURE_NAME),
                }

    @classmethod
    def mid2orig(clss, midf, include_measure_boundaries, channel):
        score = music21.converter.parse(midf)
        out = [START_OF_TRACK_NAME]
        time_signature = util.get_ts(score)
        measure_progress = 0
        measure_limit = time_signature.beatCount * time_signature.beatDuration.quarterLength
        for part in score:
            for e in part:
                if measure_progress >= measure_limit and include_measure_boundaries:
                    out.append(MEASURE_NAME)
                    measure_progress -= measure_limit
                if type(e) is music21.note.Note:
                    out.append(e.nameWithOctave if channel == 0 else e.duration.quarterLength)
                    measure_progress += e.duration.quarterLength
            break # TODO this break will only work for Nottingham-like MIDI
        out.append(END_OF_TRACK_NAME)
        return out, measure_limit

    @classmethod
    def load_from_corpus(clss, path, vocab_fname):
        if os.path.isfile(vocab_fname):
            return clss.load(vocab_fname)
        v = clss()
        filenames = util.getmidfiles(path) 
        for filename in filenames:
            for channel in range(2):
                events, _ = clss.mid2orig(filename, False, channel)
                for event in events:
                    v.add_event_to_channel(event, channel)
        print ("FactorPitchDurationVocab sizes:", v.sizes)
        v.save(vocab_fname)
        return v

    def events2mid(self, lists, out):
        s = music21.stream.Stream()
        l = zip(*lists)
        for pitch_event, duration_event in l:
            if pitch_event == self.special_events["end"] or duration_event == self.special_events["end"]:
                break
            if pitch_event in self.special_events.values() or duration_event in self.special_events.values():
                continue
            n = music21.note.Note(pitch_event.original)
            n.quarterLength = duration_event.original
            s.append(n)
            mf = music21.midi.translate.streamToMidiFile(s)
        mf.open(out, 'wb')
        mf.write()
        mf.close()


