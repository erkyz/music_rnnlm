import torch
import argparse
import time
import math, random
import torch.nn as nn
from itertools import compress, count, imap, islice
from functools import partial
from operator import eq
from torch.autograd import Variable
import sys, os
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rnnlm, rnncell_lm, hrnnlm
import data
import similarity
import util

# event index for padding
PADDING = 0

parser = argparse.ArgumentParser(description='PyTorch MIDI RNN/LSTM Language Model')

# Data stuff
parser.add_argument('--data', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--vocabf', type=str, default="../tmp/cmaj_nott_sv",
                    help='location of the saved vocabulary, or where to save it')
parser.add_argument('--corpusf', type=str, default="../tmp/cmaj_nott_sv_corpus",
                    help='location of the saved corpus, or where to save it')
parser.add_argument('--save', type=str,  default='../tmp/model.pt',
                    help='path to save the final model')

# RNN params
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--arch', type=str, default='base')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=10,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--factorize', action='store_true',
                    help='whether to factorize embeddings')

# Stuff for measure splitting
parser.add_argument('--progress_tokens', action='store_true',
                    help='whether to condition on the amount of time left until the end \
                            of the measure')
parser.add_argument('--measure_tokens', action='store_true',
                    help='whether to have a token in between measures')

# Stuff for diagonal detection
parser.add_argument('--window', type=int, default=8,
                    help='window size for note-based moving edit distances')
parser.add_argument('--c', type=int, default=2,
                    help='number of measures to base the note-based ED window off of')
parser.add_argument('--distance_threshold', type=int, default=3,
                    help='distance where below, we consider windows sufficiently similar')
parser.add_argument('--most_recent', action='store_true',
                    help='whether we repeat the most recent similar or earliest similar')

# Meta-training stuff
parser.add_argument('--skip_first_n_bar_losses', type=int, default=2, metavar='N',
                    help='"encode" first n bars')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
args = parser.parse_args()

print args

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def nth_item_index(n, item, iterable):
    if n == -1:
        return 0
    indices = compress(count(), imap(partial(eq, item), iterable))
    return next(islice(indices, n, None), -1)

def get_batch_with_conditions(source, batch, bsz, sv):
    """ Returns two Tensors corresponding to the batch """
    def pad(tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
    start_idx = batch * bsz
    this_bsz = min(bsz, (len(source) - start_idx - 1))
    source_slice = source[start_idx:start_idx+this_bsz]
    target_slice = [[mel[min(i+1,len(mel)-1)] for i in range(len(mel))] for mel in source_slice]
    maxlen = len(source_slice[0])
    data = torch.LongTensor(this_bsz,maxlen).zero_()
    conditions = torch.LongTensor(this_bsz,maxlen).zero_()
    target = torch.LongTensor(this_bsz,maxlen).zero_()
    for b in range(this_bsz):
        # TODO shouldn't be channel 0...
        melody = [sv.i2e[0][i].original for i in source_slice[b]][1:]
        batch_conditions = similarity.get_prev_match_idx(melody, args, sv)
        data[b] = pad(torch.LongTensor(source_slice[b]), maxlen)
        # We pad the end of conditions with zeros, which is technically incorrect.
        # But because the outputs are ignored anyways, we don't care.
        conditions[b] = pad(torch.LongTensor(batch_conditions), maxlen) 
        t = target_slice[b]
        '''
        second_measure_idx = nth_item_index(
            args.skip_first_n_bar_losses - 1, sv.special_events['measure'].i, t)
        for j in xrange(second_measure_idx):
            t[j] = PADDING
        '''
        target[b] = pad(torch.LongTensor(t), maxlen)
    if args.cuda: 
        data = data.cuda()
        conditions = conditions.cuda()
        target = target.cuda()
    return data, conditions, target.view(-1)

def get_batch(source, batch, bsz, sv):
    """ Returns two Tensors corresponding to the batch """
    def pad(tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
    start_idx = batch * bsz
    this_bsz = min(bsz, (len(source) - start_idx - 1))
    source_slice = source[start_idx:start_idx+this_bsz]
    target_slice = [[mel[min(i+1,len(mel)-1)] for i in range(len(mel))] for mel in source_slice]
    maxlen = len(source_slice[0])
    data = torch.LongTensor(this_bsz,maxlen).zero_()
    target = torch.LongTensor(this_bsz,maxlen).zero_()
    for i in range(this_bsz):
        data[i] = pad(torch.LongTensor(source_slice[i]), maxlen) 
        t = target_slice[i] 
        second_measure_idx = nth_item_index(
            args.skip_first_n_bar_losses - 1, sv.special_events['measure'].i, t)
        for j in xrange(second_measure_idx):
            t[j] = PADDING
        target[i] = pad(torch.LongTensor(t), maxlen)
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
    return data, target.view(-1)

def batchify(source, bsz, sv):
    batch_data = {"data": [], "targets": []}
    if args.arch == 'crnn':
        batch_data["conditions"] = []
    for channel in range(len(source)):
        channel_batches = []
        channel_targets = []
        channel_conditions = []
        for batch_idx in range(int(len(source[channel])/bsz)):
            if args.arch == 'crnn':
                data, conditions, target, = \
                    get_batch_with_conditions(source[channel], batch_idx, bsz, sv)
                channel_conditions.append(conditions)
            else:
                data, target = get_batch(source[channel], batch_idx, bsz, sv) 
            channel_batches.append(data)
            channel_targets.append(target)
        batch_data["data"].append(channel_batches)
        batch_data["targets"].append(channel_targets)
        if args.arch == 'crnn':
            batch_data["conditions"].append(channel_conditions)
    return batch_data
   
t = time.time()
if args.measure_tokens:
    args.vocabf += '_mt'
    args.corpusf += '_mt'
if args.factorize:
    if args.progress_tokens:
        vocabf = args.vocabf + '_factorized_measuretokens.p'
        corpusf = args.corpusf + '_factorized_measuretokens.p'
        sv = util.FactorPDMVocab.load_from_corpus(args.data, vocabf)
    else:
        vocabf = args.vocabf + '_factorized.p'
        corpusf = args.corpusf + '_factorized.p'
        sv = util.FactorPitchDurationVocab.load_from_corpus(args.data, vocabf)
else:
    vocabf = args.vocabf + '.p'
    corpusf = args.corpusf + '.p'
    sv = util.PitchDurationVocab.load_from_corpus(args.data, vocabf)

corpus = data.Corpus.load_from_corpus(args.data, sv, vocabf, corpusf, args.measure_tokens)
print "Time elapsed", time.time() - t

''' Size: num_channels * num_batches * num_examples_in_batch_i '''
f = '../tmp/train_batch_data_c' + str(args.c) + 'dt' + str(args.distance_threshold) + args.arch
if args.most_recent:
    f += 'recent_first'
f += '.p'
if os.path.isfile(f):
    print "Load existing train data", f
    train_data, valid_data, test_data = pickle.load(open(f, 'rb'))
else:
    print "Begin batchify"
    t = time.time()
    train_data = batchify(corpus.trains, args.batch_size, sv)
    valid_data = batchify(corpus.valids, args.batch_size, sv)
    test_data = batchify(corpus.tests, args.batch_size, sv)
    print "Saving train data to", f, "time elapsed", time.time() - t
    pickle.dump((train_data, valid_data, test_data), open(f, 'wb'))

train_mb_indices = range(0, int(len(corpus.trains[0])/args.batch_size))
valid_mb_indices = range(0, int(len(corpus.valids[0])/args.batch_size))
test_mb_indices = range(0, int(len(corpus.tests[0])/args.batch_size))

###############################################################################
# Build the model
###############################################################################

args.ntokens = sv.sizes
args.num_conditions = 4 # TODO is there a better way to do this? (hint: yes)
if args.arch == "hrnn":
    model = hrnnlm.FactorHRNNModel(args)
elif args.arch == "crnn":
    model = rnnlm.XRNNModel(args) 
else:
    model = rnnlm.RNNModel(args)
print model

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=PADDING)

###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

    return Variable(data, volatile=evaluation), Variable(target.view(-1))

def get_batch_variables(batches, batch, evaluation=False):
    ''' Size of |batches|: num_channels * num_batches * num_examples_in_batch_i '''
    batch_data = {}
    # batches is a dict
    for key in batches:
        batch_data[key] = []
    num_channels = len(batches["data"])
    for channel in range(num_channels):
        for key, value in batches.iteritems():
            batch_data[key].append(value[channel][batch])

    # Turn data into Variable_s
    batch_data_variables = {}
    for key in batch_data:
        batch_data_variables[key] = \
            [Variable(batch_data[key][c], volatile=evaluation) for c in range(num_channels)]
    return batch_data_variables

def evaluate(eval_data, mb_indices):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = corpus.vocab.sizes
    hidden = model.init_hidden(args.batch_size)
    for batch in mb_indices:
        data = get_batch_variables(eval_data, batch, evaluation=True)

        if args.arch == "hrnn":       
            data["special_event"] = corpus.vocab.special_events['measure'].i
        outputs, hidden = model(data, hidden)
        outputs_flat = [outputs[c].view(-1, ntokens[c]) for c in range(len(outputs))]
        total_loss += sum(
            [criterion(outputs_flat[c], data["targets"][c]) for c in range(len(outputs))]).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(eval_data["data"][0]) # num batches, TODO

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = corpus.vocab.sizes
    hidden = model.init_hidden(args.batch_size)
    random.shuffle(train_mb_indices)
    for batch in train_mb_indices:
        data = get_batch_variables(train_data, batch)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        if args.arch == "hrnn":       
            data["special_event"] = corpus.vocab.special_events['measure'].i
        outputs, hidden = model(data, hidden)
        outputs_flat = [outputs[c].view(-1, ntokens[c]) for c in range(len(outputs))]
        loss = sum([criterion(outputs_flat[c], data["targets"][c]) for c in range(len(outputs))])
        # loss should be a matrix, I believe.
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data
    
    return total_loss[0] / len(train_data["data"][0]) 

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train()
        val_loss = evaluate(valid_data, valid_mb_indices)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), train_loss,
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data, test_mb_indices)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

