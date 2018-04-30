from __future__ import division
import pickle, math, os
import music21
import numpy as np
from glob import glob
import fnmatch
from collections import defaultdict
import ast

import events

CONDITIONALS = {'xrnn', 'vine', 'prnn', 'ernn', 'mrnn'}
PADDING_NAME = 'padding'
START_OF_TRACK_NAME = 'start'
END_OF_TRACK_NAME = 'end'
MEASURE_NAME = 'measure'
NUM_SPLIT = 4  # number of splits per quarter note
NUM_PITCHES = 128
GEN_RESOLUTION = 480

#### Math

def normalize(x):
    denom = sum(x)
    return [i/denom for i in x]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def weightedChoice(weights, objects, apply_softmax=False, alpha=None):
    """Return a random item from objects, with the weighting defined by weights
    (which must sum to 1)."""
    if apply_softmax: weights = softmax(weights)
    if alpha: weights = normalize([w**alpha for w in weights])
    cs = np.cumsum(weights) #An array of the weights, cumulatively summed.
    idx = sum(cs < np.random.rand()) #Find the index of the first weight over a random value.
    idx = min(idx, len(objects)-1)
    return objects[idx]


#### File utils

def get_meta_dicts(path):
    if os.path.exists(path):
        return pickle.load(open(path + args.metaf, 'rb'))
    else:
        return None

def get_datadumpf(args, extra=''):
    tmp_prefix = '../tmp/' + args.tmp_prefix
    f = tmp_prefix + '_batch_data_bsz' + str(args.batch_size) + 'skip' + str(args.skip_first_n_note_losses)
    if not args.copy_earliest:
        f += '_mostrecent'
    if args.arch in CONDITIONALS:
        f += '_condmod'
    if args.arch == 'xrnn':
        f += '_xrnn'
    if args.vanilla_ckpt != '':
        f += '_vanilla'
    else:
        f += '_c' + str(args.c) + 'dt' + str(args.distance_threshold)
    f += extra + '.p'
    return f

def get_savef(args, corpus, extra=''):
    tmp_prefix = '../tmp/' 
    f = tmp_prefix + args.arch + '_batch_data_bsz' + str(args.batch_size) + 'skip' + str(args.skip_first_n_note_losses) + 'vsize' + str(corpus.vocab.sizes[0]) + 'nh' + str(args.nhid) + 'em' + str(args.emsize)
    if not args.copy_earliest:
        f += '_mostrecent'
    if args.vanilla_ckpt != '':
        f += '_vanilla'
    else:
        f += '_c' + str(args.c) + 'dt' + str(args.distance_threshold)
    f += extra + '.p'
    return f


def load_train_vocab(args):
    tmp_prefix = '../tmp/' + args.tmp_prefix
    if args.measure_tokens:
        tmp_prefix += '_mt'
    if args.factorize:
        if args.progress_tokens:
            vocabf = tmp_prefix + '_sv_factorized_measuretokens.p'
            corpusf = tmp_prefix + '_corpus_factorized_measuretokens.p'
            sv = FactorPDMVocab.load_from_corpus(args.vocab_path, vocabf)
        else:
            vocabf = tmp_prefix + '_sv_factorized.p'
            corpusf = tmp_prefix + '_corpus_factorized.p'
            sv = FactorPitchDurationVocab.load_from_corpus(args.vocab_path, vocabf)
    elif args.use_metaf and args.vocab_paths == '':
        vocabf = tmp_prefix + '_sv.p'
        corpusf = tmp_prefix + '_corpus.p'
        sv = PitchDurationVocab.load_from_pickle([args.path], vocabf)
    else:
        vocabf = tmp_prefix + '_sv.p'
        corpusf = tmp_prefix + '_corpus.p'
        sv = PitchDurationVocab.load_from_corpus(args.vocab_paths, vocabf)

    return sv, vocabf, corpusf

def itersubclasses(cls, _seen=None):
    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None: _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub

def getmidfiles(path):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.mid'):
            matches.append(os.path.join(root, filename))
    return matches

def quantize(event, resolution):
    min_pulse = resolution / NUM_SPLIT
    if event.tick % min_pulse > min_pulse/2:
        event.tick = event.tick + (min_pulse - event.tick % min_pulse)
    else:
        event.tick = event.tick - (event.tick % min_pulse)
    return event

def get_ts(score):
    # Default TimeSignature is 4/4
    time_signature = music21.meter.TimeSignature('4/4')
    for part in score:
        for e in part:
            if type(e) is music21.meter.TimeSignature:
                time_signature = e
    return time_signature


### Vocabularies

class SimpleVocab(object):
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
        ''' returns the size of each of the channel vocabularies '''
        return [len(self.events[c]) for c in range(self.num_channels)]

    def add_event_to_all(self, orig):
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
        time_signature = get_ts(score)
        measure_progress = 0
        measure_limit = time_signature.beatCount * time_signature.beatDuration.quarterLength
        for part in score:
            for e in part:
                if measure_progress >= measure_limit:
                    if include_measure_boundaries:
                        out.append((MEASURE_NAME, MEASURE_NAME))
                    measure_progress -= measure_limit
                if type(e) is music21.note.Note:
                    out.append((e.nameWithOctave, e.duration.quarterLength))
                    measure_progress += e.duration.quarterLength
                elif type(e) is music21.note.Rest:
                    out.append((e.name, e.duration.quarterLength))
                    measure_progress += e.duration.quarterLength
            if measure_progress < measure_limit:
                out.append(('rest', measure_limit - measure_progress))
            break 
        out.append((END_OF_TRACK_NAME, END_OF_TRACK_NAME))
        return out, measure_limit

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

            filenames = getmidfiles(path) 
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
        time_signature = get_ts(score)
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
        filenames = getmidfiles(path) 
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


class FactorPDMVocab(SimpleVocab):
    ''' pitch, duration, and measure progress channels '''
    ''' measure progress is how many more quarterLengths we have to go in the measure.'''
    def __init__(self):
        super(FactorPDMVocab, self).__init__(num_channels=3) 
        self.special_events = {
                "padding": self.add_event_to_all(PADDING_NAME),
                "start": self.add_event_to_all(START_OF_TRACK_NAME),
                "end": self.add_event_to_all(END_OF_TRACK_NAME),
                "measure": self.add_event_to_all(MEASURE_NAME),
                }
        self.duration_channel = 1
        self.measure_channel = 2

    @classmethod
    def mid2orig(clss, midf, include_measure_boundaries, channel):
        score = music21.converter.parse(midf)
        out = [START_OF_TRACK_NAME]
        time_signature = get_ts(score)
        measure_progress = 0
        measure_limit = time_signature.beatCount * time_signature.beatDuration.quarterLength
        for part in score:
            for e in part:
                if measure_progress >= measure_limit and include_measure_boundaries:
                    out.append(MEASURE_NAME)
                    measure_progress -= measure_limit
                if type(e) is music21.note.Note:
                    if channel == 0:
                        out.append(e.nameWithOctave)
                    elif channel == 1:
                        out.append(e.duration.quarterLength)
                    else:
                        out.append(measure_limit - measure_progress)
                    measure_progress += e.duration.quarterLength
            break # TODO this break will only work for Nottingham-like MIDI
        out.append(END_OF_TRACK_NAME)
        return out, measure_limit

    @classmethod
    def load_from_corpus(clss, path, vocab_fname):
        if os.path.isfile(vocab_fname):
            return clss.load(vocab_fname)
        v = clss()
        filenames = getmidfiles(path) 
        for filename in filenames:
            for channel in range(3):
                events, _ = clss.mid2orig(filename, False, channel)
                for event in events:
                    v.add_event_to_channel(event, channel)
        print ("FactorPDMVocab sizes:", v.sizes)
        v.save(vocab_fname)
        return v

    def events2mid(self, lists, out):
        s = music21.stream.Stream()
        l = zip(*lists)
        for pitch_event, duration_event, measure_progress in l:
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


