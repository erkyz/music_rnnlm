
import torch
import argparse
import time
import math, random
import torch.nn as nn
from torch.autograd import Variable
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchrnnlm
import data
import util

parser = argparse.ArgumentParser(description='PyTorch MIDI RNN/LSTM Language Model')

parser.add_argument('--data', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--arch', type=str, default='base')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=5,
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
parser.add_argument('--attention', action='store_true',
                    help='batch size')
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

def get_batch(source, batch, bsz):
    """ Returns two Tensors corresponding to the batch """
    def pad(tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
    start_idx = batch * bsz
    this_bsz = min(bsz, (len(source) - start_idx - 1))
    source_slice = source[start_idx:start_idx+this_bsz]
    # last element of target is wrong.
    target_slice = [[mel[min(i+1,len(mel)-1)] for i in range(len(mel))] for mel in source_slice]
    maxlen = len(source_slice[0])
    data = torch.LongTensor(this_bsz,maxlen).zero_()
    target = torch.LongTensor(this_bsz,maxlen).zero_()
    for i in range(this_bsz):
        data[i] = pad(torch.LongTensor(source_slice[i]), maxlen) 
        target[i] = pad(torch.LongTensor(target_slice[i]), maxlen)
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
    return data, target.view(-1)

def batchify(source, bsz):
    """ Returns two lists of Tensors """
    batches = []
    targets = []
    for batch in range(len(source)/bsz):
        data, target = get_batch(source, batch, bsz)
        batches.append(data)
        targets.append(target)
    return batches, targets
    
sv = util.SimpleVocab.load_from_corpus(args.data, "../tmp/cmaj_nott_sv.p")
corpus = data.Corpus(args.data, sv, args.cuda)
train_batches, train_targets = batchify(corpus.train, args.batch_size)
valid_batches, valid_targets = batchify(corpus.valid, args.batch_size)
test_batches, test_targets = batchify(corpus.test, args.batch_size)
train_mb_indices = range(0, int(len(corpus.train)/args.batch_size))
valid_mb_indices = range(0, int(len(corpus.valid)/args.batch_size))
test_mb_indices = range(0, int(len(corpus.test)/args.batch_size))

###############################################################################
# Build the model
###############################################################################

ntokens = sv.size
if args.arch == "hrnn":
    model = torchrnnlm.HRNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)
else:
    model = torchrnnlm.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.attention)
    # model = torchrnnlm.RNNCellModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.attention)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=0)

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

def get_batch_variable(batches, targets, batch, evaluation=False):
    data = batches[batch]
    target = targets[batch]
    seq_lens = [torch.nonzero(data[i]).size(0) for i in range(data.size(0))] 
    return Variable(data, volatile=evaluation), Variable(target), seq_lens

def evaluate(data_source, data_targets, mb_indices):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.vocab)
    hidden = model.init_hidden(args.batch_size)
    for batch in mb_indices:
        data, targets, seq_lens = get_batch_variable(data_source, data_targets, batch, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += criterion(output.view(-1, ntokens), targets).data 
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source) # num batches


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.vocab)
    hidden = model.init_hidden(args.batch_size)
    random.shuffle(train_mb_indices)
    for batch in train_mb_indices:
        data, targets, seq_lens = get_batch_variable(train_batches, train_targets, batch)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
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
        val_loss = evaluate(valid_batches, valid_targets, valid_mb_indices)
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
test_loss = evaluate(test_batches, test_targets, test_mb_indices)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
