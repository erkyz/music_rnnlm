
import pickle, midi, math, os
import numpy as np
from glob import glob
from collections import defaultdict

START_OF_TRACK = (-1,-1)
END_OF_TRACK = (-2,-1)
MEASURE = (-3,-1)
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
    return [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.mid'))]


#### Pitch-Duration representation

class PitchDurationEvent(object):
    def __init__(self, i, pitch, duration):
        self.i = i
        self.pitch = pitch
        self.duration = duration

    @staticmethod
    def not_found(): raise Exception("event not found in vocab")

def quantize(event, resolution):
    min_pulse = resolution / NUM_SPLIT
    if event.tick % min_pulse > min_pulse/2:
        event.tick = event.tick + (min_pulse - event.tick % min_pulse)
    else:
        event.tick = event.tick - (event.tick % min_pulse)
    return event

def mid2tuples(f):
    pattern = midi.read_midifile(f)
    resolution = pattern.resolution
    if len(pattern) < 2:
        return []
    top_melody = []
    bpm = 4
    for e in pattern[1]:
        if type(e) is midi.SetTempoEvent:
            bpm = e.get_bpm()
        if type(e) is midi.NoteOnEvent or type(e) is midi.NoteOffEvent:
            top_melody.append(e)
    out = [START_OF_TRACK]
    measure_progress = 0
    for i in range(int(len(top_melody)/2)):
        if measure_progress > NUM_SPLIT * bpm:
            out.append(MEASURE)
            measure_progress = 0
        on = top_melody[2*i]
        off = top_melody[2*i+1]
        on = quantize(on, resolution)
        off = quantize(off, resolution)
        duration = off.tick / (resolution / NUM_SPLIT)
        measure_progress += duration
        out.append((on.pitch, duration))
    out.append(END_OF_TRACK)
    return out

class SimpleVocab(object):
    def __init__(self):
        self.events = set([])
        self.tuples = set([])
        self.i2e = defaultdict(PitchDurationEvent.not_found)
        self.tup2e = defaultdict(PitchDurationEvent.not_found)
        self.special_events = {
                "start": self.add_event_tuple(START_OF_TRACK),
                "end": self.add_event_tuple(END_OF_TRACK),
                "measure": self.add_event_tuple(MEASURE),
                }
        return
      
    @property
    def size(self):
	return len(self.events)

    def __len__(self):
        return len(self.events)
 
    def add_event_tuple(self, pd):
        pitch, duration = pd
        if (pitch,duration) in self.tuples:
            return
        i = len(self.events)
        e = PitchDurationEvent(i, pitch, duration)
        self.i2e[i] = e
        self.tup2e[(pitch,duration)] = e
        self.events.add(e)
        self.tuples.add((pitch,duration))
        return e

    def __getitem__(self, key):
        if isinstance(key, int): return self.i2e[key]
        elif isinstance(key, tuple): return self.tup2e[key]
        elif isinstance(key, PitchDurationEvent): return key

    def list2mid(self, l, out):
        pattern = midi.Pattern(resolution=GEN_RESOLUTION)
        track = midi.Track()
        pattern.append(track)
        for x in l:
            e = self[x]
            if e == self.special_events["end"]:
                track.append(midi.EndOfTrackEvent())
                break
            if e in self.special_events.values():
                continue
            on = midi.NoteOnEvent(tick=0, velocity=120, pitch=e.pitch)
            track.append(on)
            off = midi.NoteOffEvent(
                    tick = e.duration * (GEN_RESOLUTION / NUM_SPLIT),
                    pitch=e.pitch)
            track.append(off)
        midi.write_midifile(out, pattern)

    def save(self, filename):
        info_dict = {
                "events": self.events,
                "tuples": self.tuples,
                "i2e": dict(self.i2e),
                "tup2e": dict(self.tup2e),
                "special_events": self.special_events,
                }
        with open(filename, "w") as f: pickle.dump(info_dict, f)

    @classmethod
    def load(clss, filename):
        with open(filename, "r") as f:
            info_dict = pickle.load(f)
            v = SimpleVocab()
            v.events = info_dict["events"]
            v.tuples = info_dict["tuples"]
            v.i2e = defaultdict(PitchDurationEvent.not_found, info_dict["i2e"])
            v.tup2e = defaultdict(PitchDurationEvent.not_found, info_dict["tup2e"])
            v.special_events = info_dict["special_events"]
            print "Vocab size:", v.size
            return v

    @classmethod
    def load_from_corpus(clss, path, vocab_fname):
        if os.path.isfile(vocab_fname):
            return SimpleVocab.load(vocab_fname)
        v = SimpleVocab()
	filenames = getmidfiles(path) 
        for filename in filenames:
            with open(filename) as f:
		events = mid2tuples(f)
		for pitch, duration in events:
		    v.add_event_tuple((pitch, duration))
        print "Vocab size:", v.size
        v.save(vocab_fname)
        return v

