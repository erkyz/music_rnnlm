from __future__ import division
import pickle, math, os
import music21
import numpy as np
import fnmatch


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

def get_meta_dicts(path, args):
    if os.path.exists(path):
        return pickle.load(open(path + args.metaf, 'rb'))
    else:
        return None

def get_datadumpf(args, extra=''):
    tmp_prefix = '../tmp/' + args.tmp_prefix
    f = tmp_prefix + '_batch_data_bsz' + str(args.batch_size) + 'skip' + str(args.skip_first_n_note_losses)
    if not args.copy_earliest:
        f += '_mostrecent'
    if args.conditional_model:
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

