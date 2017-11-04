
import os, random
import torch
import util
import pickle

class Corpus(object):
    def __init__(self, path, vocab, cuda=False):
        self.vocab = vocab
        self.train = self.eventize(os.path.join(path, 'CMaj_train'))
        self.valid = self.eventize(os.path.join(path, 'CMaj_valid'))
        self.test = self.eventize(os.path.join(path, 'CMaj_test'))

    def eventize(self, path):
        """eventizes a text file."""
        assert os.path.exists(path)

        nevents = 0
        maxlen = 0
        melodies = []
        for f in util.getmidfiles(path):
            melody = util.mid2tuples(f)
            melodies.append([self.vocab.tup2e[tup].i for tup in melody])
            maxlen = max(len(melody), maxlen)
        
        melodies.sort(key=lambda x: -len(x))
        return melodies

