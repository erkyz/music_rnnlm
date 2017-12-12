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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rnnlm, hrnnlm
import data
import util

PADDING = 0

parser = argparse.ArgumentParser(description='PyTorch MIDI RNN/LSTM Language Model')

parser.add_argument('--data', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--vocabf', type=str, default="../tmp/cmaj_nott_sv",
                    help='location of the saved vocabulary, or where to save it')
parser.add_argument('--corpusf', type=str, default="../tmp/cmaj_nott_sv_corpus",
                    help='location of the saved corpus, or where to save it')
parser.add_argument('--model', type=str, default='LSTM',
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
parser.add_argument('--skip_first_n_bar_losses', type=int, default=2, metavar='N',
                    help='"encode" first n bars')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--factorize', action='store_true',
                    help='whether to factorize embeddings')
parser.add_argument('--progress_tokens', action='store_true',
                    help='whether to condition on the amount of time left until the end \
                            of the measure')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='../tmp/model.pt',
                    help='path to save the final model')
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

def max_measure_info(source_slice):
    max_measure_length = max_num_measures = 0
    for b in source_slice:
        length_counter = 0
        num_counter = 1
        for s in b:
            length_counter += 1
            if s == sv.special_events['measure'].i:
                max_measure_length = max(max_measure_length, length_counter)
                num_counter += 1
                length_counter = 0
            max_num_measures = max(num_counter, max_num_measures)
    return max_measure_length, max_num_measures

def pad_measures(melody, measure_length, num_measures):
    measure_idxs = [0]+[nth_item_index(i, sv.special_events['measure'].i, melody) for i in range(num_measures-1)]+[len(melody)+1]
    out = []
    for i in range(len(measure_idxs)-1):
        measure = melody[measure_idxs[i]:measure_idxs[i+1]]
        out += measure
        out += [0]*(measure_length-len(measure))
    return out

def get_measure_padded_batch(source, batch, bsz, sv):
    """ Returns two Tensors corresponding to the batch, where the space between measure tokens is equal """
    start_idx = batch * bsz
    this_bsz = min(bsz, (len(source) - start_idx - 1))
    source_slice = source[start_idx:start_idx+this_bsz]
    max_measure_length, max_num_measures = max_measure_info(source_slice)
    data = torch.LongTensor(this_bsz, max_measure_length * max_num_measures).zero_()
    target = torch.LongTensor(this_bsz, max_measure_length * max_num_measures).zero_()
    measure_steps = []
    for i in range(this_bsz):
        padded = pad_measures(source_slice[i], max_measure_length, max_num_measures)
        data[i] = torch.LongTensor(padded)
        t = padded[1:] + [PADDING]
        second_measure_idx = nth_item_index(
            args.skip_first_n_bar_losses - 1, sv.special_events['measure'].i, t)
        for j in xrange(second_measure_idx):
            t[j] = PADDING
        target[i] = torch.LongTensor(t)
    data = data.cuda()
    target = target.cuda()
    for step in range(max_measure_length):
        indices = torch.LongTensor(
            [measure*max_measure_length + step for measure in range(max_num_measures)]).cuda()
        measure_steps.append(torch.index_select(data, 1, indices))
    return data, target.view(-1), measure_steps, max_measure_length, max_num_measures

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
    """ Returns three lists """
    batches = []
    targets = []
    measure_info = []
    batch_measure_steps = []
    for channel in range(len(source)):
        channel_batches = []
        channel_targets = []
        channel_measure_steps = []
        for batch in range(len(source[channel])/bsz):
            if args.arch == 'rrnn':
                data, target, measure_steps, max_measure_length, max_num_measures = \
                    get_measure_padded_batch(source[channel], batch, bsz, sv) 
            else:
                data, target = get_batch(source[channel], batch, bsz, sv) 
            if channel == 0 and args.arch == 'rrnn':
                measure_info.append((max_measure_length, max_num_measures))
            channel_batches.append(data)
            channel_targets.append(target)
            channel_measure_steps.append(measure_steps)
        batches.append(channel_batches)
        targets.append(channel_targets)
        batch_measure_steps.append(channel_measure_steps)
    return batches, targets, batch_measure_steps, measure_info
   
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
corpus = data.Corpus.load_from_corpus(args.data, sv, vocabf, corpusf)

''' Size: num_channels * num_batches * num_examples_in_batch_i '''
f = '../tmp/train_batch_data.p'
if os.path.isfile(f):
    print "Load existing train data", f
    train_batches, train_targets, train_measure_steps, train_measure_info, valid_batches, valid_targets, valid_measure_steps, valid_measure_info, test_batches, test_targets, test_measure_steps, test_measure_info = pickle.load(open(f, 'rb'))
else:
    train_batches, train_targets, train_measure_steps, train_measure_info = batchify(corpus.trains, args.batch_size, sv)
    valid_batches, valid_targets, valid_measure_steps, valid_measure_info = batchify(corpus.valids, args.batch_size, sv)
    test_batches, test_targets, test_measure_steps, test_measure_info = batchify(corpus.tests, args.batch_size, sv)
    pickle.dump((train_batches, train_targets, train_measure_steps, train_measure_info, valid_batches, valid_targets, valid_measure_steps, valid_measure_info, test_batches, test_targets, test_measure_steps, test_measure_info), open(f, 'wb'))   
 
train_mb_indices = range(0, int(len(corpus.trains[0])/args.batch_size))
valid_mb_indices = range(0, int(len(corpus.valids[0])/args.batch_size))
test_mb_indices = range(0, int(len(corpus.tests[0])/args.batch_size))

###############################################################################
# Build the model
###############################################################################

ntokens = sv.sizes
if args.arch == 'rrnn':
    model = rnnlm.RRNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)
elif args.arch == "hrnn":
    model = hrnnlm.FactorHRNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)
else:
    model = rnnlm.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)

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

def get_batch_variables(batches, targets, measure_steps, batch, evaluation=False):
    ''' Size of |batches|: num_channels * num_batches * num_examples_in_batch_i '''
    batch_data = []
    batch_target = []
    batch_measure_steps = []
    num_channels = len(batches)
    for channel in range(num_channels):
        batch_data.append(batches[channel][batch])
        batch_target.append(targets[channel][batch])
        batch_measure_steps.append(measure_steps[channel][batch])
    return [Variable(batch_data[c], volatile=evaluation) for c in range(num_channels)], [Variable(batch_target[c]) for c in range(num_channels)], [[Variable(batch_measure_steps[c][s]) for s in range(len(batch_measure_steps[c]))] for c in range(num_channels)]

def evaluate(data_source, data_targets, measure_info, measure_steps, mb_indices):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = corpus.vocab.sizes
    hidden = model.init_hidden(args.batch_size)
    for batch in mb_indices:
        data, targets, batch_measure_steps  = \
            get_batch_variables(data_source, data_targets, measure_steps, batch, evaluation=True)
        if args.arch == 'rrnn':
            measure_length, num_measures = measure_info[batch] 
            test_weights = torch.ones(args.batch_size, num_measures, num_measures).cuda() # TODO
            outputs, hidden = model(
                data, hidden, test_weights, measure_length, batch_measure_steps)
        elif args.arch == "hrnn":       
            outputs, hidden = model(data, hidden, corpus.vocab.special_events['measure'].i)
        else:
            outputs, hidden = model(data, hidden)
        outputs_flat = [outputs[i].view(-1, ntokens[i]) for i in range(len(outputs))]
        total_loss += sum(
            [criterion(outputs_flat[c], targets[c]) for c in range(len(outputs))]).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source) # num batches

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = corpus.vocab.sizes
    hidden = model.init_hidden(args.batch_size)
    random.shuffle(train_mb_indices)
    for batch in train_mb_indices:
        data, targets, measure_steps = get_batch_variables(train_batches, train_targets, train_measure_steps, batch)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        if args.arch == 'rrnn':
            measure_length, num_measures = train_measure_info[batch] 
            test_weights = torch.ones(args.batch_size, num_measures, num_measures).cuda() # TODO
            outputs, hidden = model(
                data, hidden, test_weights, measure_length, measure_steps)
        elif args.arch == "hrnn":       
            outputs, hidden = model(data, hidden, corpus.vocab.special_events['measure'].i)
        else:
            outputs, hidden = model(data, hidden)
        outputs_flat = [outputs[c].view(-1, ntokens[c]) for c in range(len(outputs))]
        loss = sum([criterion(outputs_flat[c], targets[c]) for c in range(len(outputs))])
        # loss should be a matrix, I believe.
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(corpus.train) // corpus.train_maxlen, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(valid_batches, valid_targets, valid_measure_info, valid_measure_steps, valid_mb_indices)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
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
test_loss = evaluate(test_batches, test_targets, test_measure_info, test_measure_steps, test_mb_indices)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

