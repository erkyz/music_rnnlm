
import os, random
import torch
import util
import pickle

MAX_ALLOWED_MULTIPLIER = 2

class Corpus(object):
    def __init__(self, path, vocab):
        self.vocab = vocab
        self.train = self.eventize(os.path.join(path, 'train'))
        self.train_mb_indices = 
        self.valid = self.eventize(os.path.join(path, 'valid'))
        self.test = self.eventize(os.path.join(path, 'test'))

    def eventize(self, path):
        """eventizes a text file."""
        assert os.path.exists(path)

        events = []
        for f in files:
            tuples = util.mid2tuples(f)
            events.append(self.vocab.tup2e[tup].i)
        events.sort(key=lambda x:-len(x))

        print "="*50
        print path
        print "Num files", len(files)
        print "="*50
        return events


