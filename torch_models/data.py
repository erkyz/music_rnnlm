
import os, random
import torch
import util
import pickle
import similarity

class Corpus(object):
    def __init__(self):
        self.vocab = None
        self.vocab_fname = ""
        self.my_fname = ""
        self.trains = [[]]
        self.valids = [[]]
        self.tests = [[]]

    def eventize(self, path, args):
        ''' returns a list of lists, where each list is a single channel. '''
        assert os.path.exists(path)

        nevents = 0
        maxlen = 0
        melodies = [[] for _ in range(self.vocab.num_channels)]
        for f in util.getmidfiles(path):
            for c in range(self.vocab.num_channels):
                melody, _ = self.vocab.mid2orig(f, include_measure_boundaries=args.measure_tokens, channel=c)
                if len(melody) < 10 or len(melody) > 400:
                    continue
                melody2, _ = self.vocab.mid2orig(f, include_measure_boundaries=True, channel=c)
                melody2 = melody2[1:]
                melodies[c].append(
                    (
                        [self.vocab.orig2e[c][orig].i for orig in melody],
                        max(int(args.c*similarity.get_avg_dist_between_measures(melody2, self.vocab)), similarity.MIN_WINDOW)
                    )
                        )
        for c in range(self.vocab.num_channels):
            melodies[c].sort(key=lambda x: -len(x[0]))
        return melodies

    def save(self):
        info_dict = {
                "trains": self.trains,
                "valids": self.valids,
                "tests": self.tests,
                "vocab_fname": self.vocab_fname,
                }
        with open(self.my_fname, "w") as f: pickle.dump(info_dict, f)
        self.vocab.save(self.vocab_fname)

    @classmethod
    def load(clss, filename):
        with open(filename, "r") as f:
            info_dict = pickle.load(f)
        corpus = clss()
        print "Load", corpus
        corpus.trains = info_dict["trains"]
        corpus.valids = info_dict["valids"]
        corpus.tests = info_dict["tests"]
        corpus.vocab_fname = info_dict["vocab_fname"]
        corpus.vocab = util.PitchDurationVocab.load(corpus.vocab_fname)
        return corpus

    @classmethod
    def load_from_corpus(clss, path, vocab, vocab_fname, corpus_fname, args):
        if os.path.isfile(corpus_fname):
            print "Loading existing Corpus", corpus_fname
            return clss.load(corpus_fname)
        print "Creating new Corpus", corpus_fname
        corpus = clss()
        corpus.vocab = vocab
        corpus.vocab_fname = vocab_fname
        corpus.my_fname = corpus_fname
        corpus.trains = corpus.eventize(os.path.join(path, 'train'), args)
        corpus.valids = corpus.eventize(os.path.join(path, 'valid'), args)
        corpus.tests = corpus.eventize(os.path.join(path, 'test'), args)

        print "Saving new Corpus", corpus_fname
        corpus.save()
        return corpus



