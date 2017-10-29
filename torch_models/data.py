
import os, random
import torch
import util
import pickle

MAX_ALLOWED_MULTIPLIER = 2

class Corpus(object):
    def __init__(self, path, dictionary):
        self.dictionary  = dictionary
        self.train, self.train_masks, self.train_maxlen = \
                self.eventize(os.path.join(path, 'train'))
        self.valid, self.valid_masks, self.valid_maxlen = \
                self.eventize(os.path.join(path, 'valid'))
        self.test, self.test_masks, self.test_maxlen = \
                self.eventize(os.path.join(path, 'test'))

    def eventize(self, path):
        """eventizes a text file."""
        assert os.path.exists(path)

        nevents = 0
        files = []
        for f in util.getmidfiles(path):
            melody = util.mid2tuples(f)
            files.append(f)
            nevents += len(melody) + 1
        avg_len = nevents / len(files)
        max_allowed_len = avg_len * MAX_ALLOWED_MULTIPLIER

        random.shuffle(files)

        # mildly unncessary
        num_within_range = sum(
                [1 if len(util.mid2tuples(f)) <= max_allowed_len else 0 for f in files])
        
        ids = torch.LongTensor(num_within_range * max_allowed_len)
        masks = torch.FloatTensor(num_within_range * max_allowed_len)
        event_num = 0
        for f in files:
            tuples = util.mid2tuples(f)
            if len(tuples) > max_allowed_len:
                # TODO this will NOT scale to larger datasets
                continue
            for tup in tuples:
                ids[event_num] = self.dictionary.tup2i[tup]
                masks[event_num] = 1
                event_num += 1
            for i in range(max_allowed_len - len(tuples)):
                ids[event_num] = self.dictionary.e2i[self.dictionary.special_events["end"]]
                masks[event_num] = 0
                event_num += 1

        print "="*50
        print path
        print "Num files", len(files)
        print "Threw out", len(files) - num_within_range
        print "BPPT", max_allowed_len
        print "="*50
        return ids, masks, max_allowed_len, 


