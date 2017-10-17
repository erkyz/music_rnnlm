
import os
import torch
import util

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

        def add_word(self, word):
            if word not in self.word2idx:
                self.idx2word.append(word)
        self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

        def __len__(self):
            return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, dictionary):
        self.dictionary  = dictionary
        self.train = self.eventize(os.path.join(path, 'train'))
        self.valid = self.eventize(os.path.join(path, 'valid'))
        self.test = self.eventize(os.path.join(path, 'test'))

    def eventize(self, path):
        """eventizes a text file."""
        assert os.path.exists(path)

        num_events = 0
        for f in util.getmidfiles(path):
            melody = util.mid2tuples(f)
            num_events += len(melody)

        ids = torch.LongTensor(num_events)
        event_num = 0
        for f in util.getmidfiles(path):
            tuples = util.mid2tuples(f)
            for tup in tuples:
                ids[event_num] = self.dictionary.tup2i[tup]
                event_num += 1

        return ids

