import torch
import argparse
import time
import math, random
import torch.nn as nn
from itertools import compress, count, imap, islice
from functools import partial
from operator import eq
from torch.autograd import Variable
import torch.nn.functional as F
import sys, os
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import util
import data
import rnnlm, rnncell_lm, hrnnlm
import similarity
import get_hiddens, old_generate

# event index for padding
PADDING = 0

parser = argparse.ArgumentParser(description='PyTorch MIDI RNN/LSTM Language Model')

# Data stuff
parser.add_argument('--data', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--tmp_prefix', type=str, default="cmaj_nott",
                    help='tmp directory + prefix for tmp files')
parser.add_argument('--save', type=str, default="",
                    help='override default model save filename')

# RNN params
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--arch', type=str, default='base')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
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
parser.add_argument('--vanilla_ckpt', type=str,  default='',
                    help='pretrained vanilla model dir')
parser.add_argument('--c', type=float, default=2,
                    help='number of measures to base the note-based ED window off of')
parser.add_argument('--distance_threshold', type=int, default=3,
                    help='distance where below, we consider windows sufficiently similar')
parser.add_argument('--most_recent', action='store_true',
                    help='whether we repeat the most recent similar or earliest similar')

# "Get hiddens" / Generate stuff
parser.add_argument('--max_events', type=int, default='250',
                    help='number of words to generate')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--condition_piece', type=str, default="",
                    help='midi piece to condition on')
parser.add_argument('--checkpoint', type=str, default='../tmp/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--num_out', type=int, default=5,
                    help='number of melodies to generate')
parser.add_argument('--condition_notes', type=int, default=0,
                    help='number of notes to condition the generation on')
parser.add_argument('--outf', type=str, default='test',
                    help='output file for generated text')


# Meta-training stuff
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--skip_first_n_note_losses', type=int, default=0,
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
# Helper functions
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch_with_conditions(source, batch, bsz, sv, vanilla_model=None):
    """ Returns two Tensors corresponding to the batch """
    def pad(tensor, length, val):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_() + val])
    start_idx = batch * bsz
    this_bsz = min(bsz, (len(source) - start_idx)) # TODO why was there a -1 here?
    source_slice = source[start_idx:start_idx+this_bsz]
    target_slice = [[mel[i+1] for i in range(len(mel)-1)] + [PADDING] for mel, _ in source_slice]
    maxlen = len(source_slice[0][0])
    data = torch.LongTensor(this_bsz,maxlen).zero_()
    conditions = torch.LongTensor(this_bsz,maxlen).zero_()
    target = torch.LongTensor(this_bsz,maxlen).zero_()
    for b in range(this_bsz):
        # TODO shouldn't be channel 0...
        mel_idxs = source_slice[b][0]
        if args.vanilla_ckpt == '':
            melody = [sv.i2e[0][i].original for i in mel_idxs][1:] # remove START
            args.window = source_slice[b][1]
            ssm, _ = similarity.get_note_ssm_future(melody, args, bnw=True)
        else:
            ssm = similarity.get_rnn_ssm(args, sv, vanilla_model, [mel_idxs])
        batch_conditions = similarity.get_prev_match_idx(ssm, args, sv)
        # print zip(mel_idxs, range(len(mel_idxs)))
        # print zip(batch_conditions, range(len(batch_conditions)))
        data[b] = pad(torch.LongTensor(mel_idxs), maxlen, PADDING)
        # We pad the end of conditions with zeros, which is technically incorrect.
        # But because the outputs are ignored anyways, we don't care.
        conditions[b] = pad(torch.LongTensor(batch_conditions), maxlen, -1) 
        t = target_slice[b] 
        for j in xrange(min(args.skip_first_n_note_losses, len(t))):
            t[j] = PADDING
        target[b] = pad(torch.LongTensor(t), maxlen, PADDING)
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
    this_bsz = min(bsz, (len(source) - start_idx))
    source_slice = source[start_idx:start_idx+this_bsz]
    target_slice = [[mel[i+1] for i in range(len(mel)-1)] + [PADDING] for mel, _ in source_slice]
    maxlen = len(source_slice[0][0])
    print maxlen
    print source_slice
    print target_slice
    data = torch.LongTensor(this_bsz,maxlen).zero_()
    target = torch.LongTensor(this_bsz,maxlen).zero_()
    for i in range(this_bsz):
        data[i] = pad(torch.LongTensor(source_slice[i][0]), maxlen) 
        t = target_slice[i] 
        for j in xrange(args.skip_first_n_note_losses):
            t[j] = PADDING
        target[i] = pad(torch.LongTensor(t), maxlen)
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
    return data, target.view(-1)

def batchify(source, bsz, sv, vanilla_model):
    batch_data = {"data": [], "targets": []}
    if args.arch in util.CONDITIONALS:
        batch_data["conditions"] = []
    for channel in range(len(source)):
        channel_batches = []
        channel_targets = []
        channel_conditions = []
        for batch_idx in range(int(len(source[channel])/bsz)):
            # For each batch, create a Tensor
            if args.arch in util.CONDITIONALS:
                data, conditions, target, = \
                    get_batch_with_conditions(source[channel], batch_idx, bsz, sv, vanilla_model)
                channel_conditions.append(conditions)
            else:
                data, target = get_batch(source[channel], batch_idx, bsz, sv) 
            channel_batches.append(data)
            channel_targets.append(target)
        batch_data["data"].append(channel_batches)
        batch_data["targets"].append(channel_targets)
        if args.arch in util.CONDITIONALS:
            batch_data["conditions"].append(channel_conditions)
    return batch_data

def get_batch_variables(batches, batch, evaluation=False):
    # Used because you can't save Variables with pickle
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
    variable_batch_data = {}
    for key in batch_data:
        variable_batch_data[key] = \
            [Variable(batch_data[key][c], volatile=evaluation) for c in range(num_channels)]
    variable_batch_data["cuda"] = args.cuda 
    return variable_batch_data


###############################################################################
# Build the model
###############################################################################

t = time.time()
sv, vocabf, corpusf = util.load_train_vocab(args)
corpus = data.Corpus.load_from_corpus(args.data, sv, vocabf, corpusf, args)
print "Time elapsed", time.time() - t

args.ntokens = sv.sizes
vanilla_model = None

if args.mode == 'train':
    if args.arch == "hrnn":
        model = hrnnlm.FactorHRNNModel(args)
    elif args.arch == "xrnn":
        if args.vanilla_ckpt != '':
            with open(args.vanilla_ckpt, 'rb') as f:
                vanilla_model = torch.load(f)
                # vanilla_model.eval()
        model = rnncell_lm.XRNNModel(args) 
    elif args.arch == "cell":
        model = rnncell_lm.RNNCellModel(args) 
    elif args.arch == "vine":
        if args.vanilla_ckpt != '':
            with open(args.vanilla_ckpt, 'rb') as f:
                vanilla_model = torch.load(f)
        model = rnncell_lm.VineRNNModel(args) 
    else:
        model = rnnlm.RNNModel(args)

    criterion = nn.CrossEntropyLoss(ignore_index=PADDING)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

elif args.mode == 'get_hiddens' or args.mode == 'generate':
    args.batch_size = 1
    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f)
        model.eval()
    if args.vanilla_ckpt != '':
        with open(args.vanilla_ckpt, 'rb') as f:
            vanilla_model = torch.load(f)
else:
    print "Mode not supported."
    sys.exit()

print model

if args.cuda:
    model.cuda()
else:
    model.cpu()


###############################################################################
# Get the data
###############################################################################

''' Size: num_channels * num_batches * num_examples_in_batch_i '''
f = util.get_datadumpf(args)
if os.path.isfile(f):
    print "Load existing train data", f
    train_data, valid_data, test_data = pickle.load(open(f, 'rb'))
else:
    print "Begin batchify"
    t = time.time()
    train_data = batchify(corpus.trains, args.batch_size, sv, vanilla_model)
    valid_data = batchify(corpus.valids, args.batch_size, sv, vanilla_model)
    test_data = batchify(corpus.tests, args.batch_size, sv, vanilla_model)
    print "Saving train data to", f, "time elapsed", time.time() - t
    pickle.dump((train_data, valid_data, test_data), open(f, 'wb'))

train_mb_indices = range(0, int(len(corpus.trains[0])/args.batch_size))
valid_mb_indices = range(0, int(len(corpus.valids[0])/args.batch_size))
test_mb_indices = range(0, int(len(corpus.tests[0])/args.batch_size))

###############################################################################
# Training code
###############################################################################

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
        outputs, hidden = model(data, hidden, args)
        outputs_flat = [outputs[c].view(-1, ntokens[c]) for c in range(len(outputs))]
        total_loss += sum(
            [criterion(outputs_flat[c], data["targets"][c]) for c in range(len(outputs))]).data
        # hidden = repackage_hidden(hidden)
        hidden = model.init_hidden(args.batch_size)
    return total_loss[0] / len(mb_indices)

# TODO get rid of the x param
def train(x):
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
        # hidden = repackage_hidden(hidden)
        hidden = model.init_hidden(args.batch_size)
        if args.arch == "hrnn":       
            data["special_event"] = corpus.vocab.special_events['measure'].i
        outputs, hidden = model(data, hidden, args)
        outputs_flat = [outputs[c].view(-1, ntokens[c]) for c in range(len(outputs))]
        loss = sum([criterion(outputs_flat[c], data["targets"][c]) for c in range(len(outputs))])
        # TODO with multiple channels, this is a multiple of batch_size
        optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        '''
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)
        '''
        total_loss += loss.data
 
    # divide by number of batches since we're cumulating the losses for each batch
    return total_loss[0] / len(train_mb_indices) 

# Loop over epochs.
lr = args.lr
best_val_loss = None
losses = {'train': [], 'valid': []}

if args.mode == 'train':
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            args.epoch = epoch-1
            epoch_start_time = time.time()
            x = (epoch == 100)
            train_loss = train(x) # TODO
            val_loss = evaluate(valid_data, valid_mb_indices)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), train_loss,
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            losses["train"].append(train_loss)
            losses["valid"].append(val_loss)
            pickle.dump(losses, open(util.get_datadumpf(args, extra='curves'), 'wb'))
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save if args.save != '' else util.get_savef(args, corpus), 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save if args.save != '' else util.get_savef(args, corpus), 'rb') as f:
        print("Saving model")
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data, test_mb_indices)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

elif args.mode == 'generate':
    old_generate.generate(model, args, corpus.vocab)

elif args.mode == 'get_hiddens':
    get_hiddens.get_hiddens(model, args, corpus.vocab)


