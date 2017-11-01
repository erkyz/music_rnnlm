
import os, random
import torch
import util
import pickle

class Corpus(object):
    def __init__(self, path, vocab, cuda=False):
        self.vocab = vocab
        '''
        self.train, self.train_maxlen = self.eventize(os.path.join(path, 'train'))
        self.valid, self.vlaid_maxlen = self.eventize(os.path.join(path, 'valid'))
        self.test, self.test_maxlen = self.eventize(os.path.join(path, 'test'))
        ''' 
        self.train = self.eventize(os.path.join(path, 'train'))
        self.valid = self.eventize(os.path.join(path, 'valid'))
        self.test = self.eventize(os.path.join(path, 'test'))

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
        
        ids = torch.LongTensor(len(melodies) * maxlen)
        melodies.sort(key=lambda x: -len(x))
        return melodies

	'''
	print len(melodies[0])
	print len(melodies[-1])
        event_num = 0
        for melody in melodies:
            for tup in melody:
                ids[event_num] = self.vocab.tup2e[tup].i
                event_num += 1
            for i in range(maxlen - len(melody)):
                ids[event_num] = self.vocab.special_events["end"].i
                event_num += 1

        print "="*50
        print path
        print "Num files", len(melodies)
        print "="*50
	return ids, maxlen
	'''
