import torch
import argparse
import time
import math, random
import csv
import torch.nn as nn
from itertools import compress, count, imap, islice, combinations
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
import get_hiddens, old_generate, gen_util

# event index for padding
PADDING = 0

parser = argparse.ArgumentParser(description='PyTorch MIDI RNN/LSTM Language Model')

# Data stuff
parser.add_argument('--path', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--vocab_paths', type=str, default='',
                    help='location of the data corpus used for the vocabulary')
parser.add_argument('--tmp_prefix', type=str, default="cmaj_nott",
                    help='tmp directory + prefix for tmp files')
parser.add_argument('--save', type=str, default="",
                    help='override default model save filename')
parser.add_argument('--train_info_out', type=str, default="test.csv",
                    help='where to save train info')
parser.add_argument('--use_metaf', action='store_true')
parser.add_argument('--synth_data', action='store_true')

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
parser.add_argument('--ss', action='store_true',
                    help='scheduled sampling')
parser.add_argument('--baseline', action='store_true',
                    help='use baseline version of model')


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
parser.add_argument('--copy_earliest', action='store_true',
                    help='whether we repeat the most recent similar or earliest similar')

# "Get hiddens" / Generate stuff
parser.add_argument('--max_events', type=int, default='250',
                    help='number of words to generate')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--condition_piece', type=str, default="",
                    help='midi piece to condition on')
parser.add_argument('--checkpoint', type=str, default='',
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

def get_and_save_rnn_ssms(source, sv, vanilla_model, save_dir):
    ssms = {}
    for mel_idxs, metadata in source[0]:
        ssm = similarity.get_rnn_ssm(args, sv, vanilla_model, [mel_idxs])
        ssms[metadata['f']] = ssm
    pickle.dump(ssms, open(args.path + save_dir + '/ssms.p', 'wb'))

def get_batch_metadata(source, batch, bsz):
    start_idx = batch * bsz
    this_bsz = min(bsz, (len(source) - start_idx)) # TODO why was there a -1 here?
    source_slice = source[start_idx:start_idx+this_bsz]
    metadata = []
    for b in range(this_bsz):
        print source_slice[b][1]['segments']
        metadata.append(source_slice[b][1]['segments'])
    return metadata


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
    conditions = [None for b in range(bsz)]
    target = torch.LongTensor(this_bsz,maxlen).zero_()
    for b in range(this_bsz):
        # TODO shouldn't be channel 0...
        mel_idxs = source_slice[b][0]
        if args.use_metaf:
            batch_conditions = source_slice[b][1]['segment_sdm']
            # print source_slice[b][1]['segments']
            # print source_slice[b][1]['f']
            # print ssm
            # print maxlen
        elif args.vanilla_ckpt == '':
            pass
            # TODO other similarity measures.
        else:
            ssm = similarity.get_rnn_ssm(args, sv, vanilla_model, [mel_idxs])
        '''
        else:
            batch_conditions = similarity.get_prev_match_idx(ssm, args, sv)
        '''
        # print zip(mel_idxs, range(len(mel_idxs)))
        # print zip(batch_conditions, range(len(batch_conditions)))
        data[b] = pad(torch.LongTensor(mel_idxs), maxlen, PADDING)
        conditions[b] = batch_conditions
        t = target_slice[b] 
        for j in xrange(min(args.skip_first_n_note_losses, len(t))):
            t[j] = PADDING
        target[b] = pad(torch.LongTensor(t), maxlen, PADDING)
    if args.cuda: 
        data = data.cuda()
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
    batch_data = {"data": [], "targets": [], "metadata": []}
    if args.arch in util.CONDITIONALS:
        batch_data["conditions"] = []
    for channel in range(len(source)):
        channel_batches = []
        channel_targets = []
        channel_metadata = []
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
            if args.use_metaf:
                channel_metadata.append(get_batch_metadata(source[channel], batch_idx, bsz))
            else:
                channel_metadata.append([])
        batch_data["data"].append(channel_batches)
        batch_data["targets"].append(channel_targets)
        batch_data["metadata"].append(channel_metadata)
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
        if key in {'data', 'targets'}:
            variable_batch_data[key] = \
                [Variable(batch_data[key][c], volatile=evaluation) for c in range(num_channels)]
        else:
            variable_batch_data[key] = batch_data[key]
            
    variable_batch_data["cuda"] = args.cuda 
    return variable_batch_data


###############################################################################
# Build the model
###############################################################################

t = time.time()
sv, vocabf, corpusf = util.load_train_vocab(args)
corpus = data.Corpus.load_from_corpus(sv, vocabf, corpusf, args)
print "Time elapsed", time.time() - t

args.ntokens = sv.sizes
vanilla_model = None

if args.mode == 'train':
    if args.checkpoint != '':
        print "Loading model checkpoint"
        with open(args.checkpoint, 'rb') as f:
            model = torch.load(f)
    elif args.arch == "hrnn":
        model = hrnnlm.FactorHRNNModel(args)
    elif args.arch == "prnn":
        model = rnncell_lm.PRNNModel(args)
    elif args.arch == "xrnn":
        model = rnncell_lm.XRNNModel(args) 
    elif args.arch == "ernn":
        model = rnncell_lm.ERNNModel(args) 
    elif args.arch == "mrnn":
        model = rnncell_lm.MRNNModel(args)
        args.out2 = True
    elif args.arch == "cell":
        model = rnncell_lm.RNNCellModel(args) 
    elif args.arch == "vine":
        model = rnncell_lm.VineRNNModel(args) 
    else:
        model = rnnlm.RNNModel(args)
    
    if args.arch in util.CONDITIONALS:
        if args.vanilla_ckpt != '':
            with open(args.vanilla_ckpt, 'rb') as f:
                vanilla_model = torch.load(f)
                # vanilla_model.eval()

    sigmoid = nn.Sigmoid()
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print model

elif args.mode in ['get_hiddens', 'generate']:
    checkpoint = args.checkpoint if args.checkpoint != '' else util.get_savef(args, corpus)
    args.batch_size = 1
    with open(checkpoint, 'rb') as f:
        model = torch.load(f)
        model.eval()
    if args.vanilla_ckpt != '':
        with open(args.vanilla_ckpt, 'rb') as f:
            vanilla_model = torch.load(f)
            vanilla_model.eval()

if args.mode not in ['get_rnn_ssms']:
    if args.cuda:
        model.cuda()
    else:
        model.cpu()


###############################################################################
# Get the data
###############################################################################

if args.mode == 'train':
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

NUM_TO_GENERATE = 50

def evaluate_ssm():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    path = args.path + 'train/'
    meta_dicts = util.get_meta_dicts(path)
    num_total_repeats = 0
    total_repeat_ed = 0
    for songf, info in random.sample(meta_dicts.items(), NUM_TO_GENERATE):
        args.condition_piece = path + songf
        meta_dict = meta_dicts[os.path.basename(songf)]

        conditions = []
        events = gen_util.get_events(sv, args, args.condition_piece, meta_dict)
        if args.arch in util.CONDITIONALS:
            conditions = gen_util.get_conditions(sv, args, vanilla_model, meta_dict)

        generated = old_generate.generate(model, events, conditions, meta_dict, args, corpus.vocab,
                vanilla_model)
        print [e.i for e in generated]
        print conditions
        gen_measure_sdm = similarity.get_measure_sdm(
                [e.original for e in generated[1:][:-1]], 
                meta_dict['segments'])
        repeating_sections = meta_dict['repeating_sections']
        for repeats in repeating_sections:
            repeat_tups = list(combinations(repeats, 2))
            for i,j in repeat_tups:
                total_repeat_ed += gen_measure_sdm[i,j]
            num_total_repeats += len(repeat_tups)

        '''
        gold_measure_ssm = similarity.get_measure_ssm(
                [e.original for e in [sv.i2e[0][i] for i in events[0]][1:][:-1]], 
                meta_dict['segments'], args)
        '''
    return total_repeat_ed / num_total_repeats


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

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = corpus.vocab.sizes
    hidden = model.init_hidden(args.batch_size)
    random.shuffle(train_mb_indices)
    if args.arch in util.CONDITIONALS:
        print torch.sum(model.A).data[0]
        print torch.sum(model.B).data[0]
    for batch in train_mb_indices:
        data = get_batch_variables(train_data, batch)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # hidden = repackage_hidden(hidden)
        hidden = model.init_hidden(args.batch_size)
        if args.arch == "hrnn":       
            data["special_event"] = corpus.vocab.special_events['measure'].i
        outputs, hidden = model(data, hidden, args)
        '''
        print data["targets"][0]
        print data["conditions"]
        print outputs[0][0,11]
        assert(False)
        '''

        # print outputs
        word_idxs = []
        for i in range(outputs[0].size(1)):
            m, am = 0, 0
            for j in range(outputs[0].size(2)):
                if outputs[0][0,i,j].data[0] > m:
                    am = j
                    m = outputs[0][0,i,j].data[0]
            word_idxs.append(am)
        '''
        print data["data"][0]
        print data["conditions"]
        '''

        outputs_flat = [outputs[c].view(-1, ntokens[c]) for c in range(len(outputs))]
        loss = sum([criterion(outputs_flat[c], data["targets"][c]) for c in range(len(outputs))])
        # TODO with multiple channels, this is a multiple of batch_size
        optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.data
 
    # divide by number of batches since we're cumulating the losses for each batch
    return total_loss[0] / len(train_mb_indices) 

# Loop over epochs.
lr = args.lr
best_val_loss = None
losses = {'train': [], 'valid': []}
train_outf = open(args.train_info_out, 'wb')
writer = csv.writer(train_outf, delimiter=',')
writer.writerow(['train_loss','val_loss','gen_ED'])

if args.mode == 'train':
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            args.epoch = epoch-1
            epoch_start_time = time.time()
            train_loss = train()
            if args.synth_data:
                gen_ED = evaluate_ssm()
                print gen_ED
            val_loss = evaluate(valid_data, valid_mb_indices)
            val_perp = math.exp(val_loss) if val_loss < 100 else float('nan')
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), train_loss,
                                               val_loss, val_perp))
            print('-' * 89)
            losses["train"].append(train_loss)
            losses["valid"].append(val_loss)
            if args.use_metaf and args.synth_data:
                writer.writerow([train_loss,val_loss,gen_ED])
            else:
                writer.writerow([train_loss,val_loss])
            # Write to file without closing
            train_outf.flush()
            os.fsync(train_outf.fileno())
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
        train_outf.close()
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
    train_outf.close()

elif args.mode == 'generate':
    for i in range(args.num_out):
        torch.manual_seed(i*args.seed)
        path = args.path + 'train'
        if args.use_metaf:
            meta_dicts = util.get_meta_dicts(path)
            meta_dict = meta_dicts[os.path.basename(args.condition_piece)]
        else:
            meta_dict = None

        conditions = []
        # Note: |events| is only for conditions
        events = gen_util.get_events(sv, args, args.condition_piece, meta_dict)
        if args.arch in util.CONDITIONALS:
            conditions = gen_util.get_conditions(sv, args, vanilla_model, meta_dict)

        generated = old_generate.generate(model, events, conditions, meta_dict, args, corpus.vocab, 
                vanilla_model, end=True)
        outf = "../../generated/" + args.outf + '_' + str(i) + '.mid'
        sv.events2mid([generated], outf)

elif args.mode == 'get_hiddens':
    args.epoch = 0
    get_hiddens.get_hiddens(model, args, corpus.vocab, vanilla_model)

elif args.mode == 'get_rnn_ssms':
    # for song in the directory, get RNN SSM
    with open(args.vanilla_ckpt, 'rb') as f:
        vanilla_model = torch.load(f)
        vanilla_model.eval()
    get_and_save_rnn_ssms(corpus.trains, sv, vanilla_model, 'train')
    get_and_save_rnn_ssms(corpus.valids, sv, vanilla_model, 'valid')
    get_and_save_rnn_ssms(corpus.tests, sv, vanilla_model, 'test')




