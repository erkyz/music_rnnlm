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

from utils import *
from constants import *
from vocab import *
import util
import corpus
import similarity
import generate, gen_util

parser = argparse.ArgumentParser(description='PyTorch MIDI RNN/LSTM Language Model')

# Meta-training stuff
parser.add_argument('--mode', type=str, 
                    help='one of (train, generate, get_hiddens)')
parser.add_argument('--skip_first_n_note_losses', type=int, default=0,
                    help='"encode" first n bars')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

# Data stuff
parser.add_argument('--path', type=str, default='music_data/CMaj_Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--vocab_paths', type=str, default='',
                    help='list (in string form) of location of the data corpuses
                        used for the vocabulary')
parser.add_argument('--tmp_prefix', type=str, default="tmp",
                    help='prefix for tmp files')
parser.add_argument('--save', type=str, default="",
                    help='override default model save filename')
parser.add_argument('--train_info_out', type=str, default="test.csv",
                    help='where to save train info')
parser.add_argument('--metaf', type=str, 
                    help='name of metadata file, e.g. "meta.p"')
parser.add_argument('--synth_data', action='store_true',
                    help='if we use synthetic data. has some effects on data processing')

# RNN params
# TODO support RNN_TANH, RNN_RELU, LSTM (should be easy)
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--arch', type=str, default='base',
                    help='currently, one of (base, cell, attn, readrnn)')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--gates_nhid', type=int, default=256,
                    help='number of hidden units for any gates')
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
parser.add_argument('--factorize', action='store_true',
                    help='whether to factorize embeddings')
parser.add_argument('--ss', action='store_true',
                    help='use scheduled sampling')
parser.add_argument('--input_feed_num', type=int, default=0, 
                    help='number of future measures to feed')
parser.add_argument('--input_feed_dim', type=int, default=0, 
                    help='num dimensions to concat for input feeding')

# CNN params
parser.add_argument('--cnn_encoder', action='store_true',
                    help='use a CNN to encode the SSM that the RNN decodes')

# Stuff for measure splitting. I haven't used this in a while.
parser.add_argument('--measure_tokens', action='store_true',
                    help='whether to include a separate token in between measures')

# "Get hiddens" / Generate stuff
parser.add_argument('--num_out', type=int, default=5,
                    help='number of melodies to generate')
parser.add_argument('--max_events', type=int, default='250',
                    help='number of words to generate per example')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature -- higher will increase diversity')
parser.add_argument('--condition_piece', type=str, default="",
                    help='midi piece to condition on')
parser.add_argument('--checkpoint', type=str, default='',
                    help='model checkpoint to use')
parser.add_argument('--condition_notes', type=int, default=0,
                    help='number of notes to condition the generation on')
parser.add_argument('--outf', type=str, default='test',
                    help='output file for generated text')

args = parser.parse_args()
# Everything uses metaf now, but still may want to do something else later.
args.use_metaf = (args.metaf != '')

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

def load_train_vocab(args):
    # Save all tmp files in the tmp/ directory (must exist)
    tmp_prefix = 'tmp/' + args.tmp_prefix
    if args.factorize:
        vocabf = tmp_prefix + '_sv_factorized.p'
        corpusf = tmp_prefix + '_corpus_factorized.p'
        sv = FactorPitchDurationVocab.load_from_corpus(args.vocab_path, vocabf)
    elif args.use_metaf and args.vocab_paths == '':
        vocabf = tmp_prefix + '_sv.p'
        corpusf = tmp_prefix + '_corpus.p'
        sv = PitchDurationVocab.load_from_pickle([args.path], vocabf)
    else:
        vocabf = tmp_prefix + '_sv.p'
        corpusf = tmp_prefix + '_corpus.p'
        sv = PitchDurationVocab.load_from_corpus(args.vocab_paths, vocabf)

    # The vocab has been saved to |vocobf|, likewise for the corpus.
    return sv, vocabf, corpusf

def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    I don't think this is necessary, but just in case.
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch_metadata(source, batch, bsz):
    start_idx = batch * bsz
    this_bsz = min(bsz, (len(source) - start_idx)) # TODO why was there a -1 here?
    source_slice = source[start_idx:start_idx+this_bsz]
    metadata = []
    for b in range(this_bsz):
        metadata.append(source_slice[b][1]['measure_boundaries'])
    return metadata

def create_batch_tensor_with_conds(source, batch_idx, bsz, sv):
    """ 
    Returns data/condition/target Tensors corresponding to the batch starting at |batch_idx| 
        that's |bsz| long
    """
    def pad(tensor, length, val):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_() + val])
    start_idx = batch_idx * bsz
    this_bsz = min(bsz, (len(source) - start_idx))
    source_slice = source[start_idx:start_idx+this_bsz]
    # Just predict PADDING_IDX after END. We ignore all padding in loss.
    target_slice = [[mel[i+1] for i in range(len(mel)-1)] + [PADDING_IDX] for mel, _ in source_slice]
    # Source is sorted by length (in corpus.py), so this is the maximum-length in 
    # this batch. We will make the batch this length and pad the rest.
    maxlen = len(source_slice[0][0])
    data = torch.LongTensor(this_bsz,maxlen).zero_()
    conditions = [None for b in range(bsz)]
    target = torch.LongTensor(this_bsz,maxlen).zero_()
    for b in range(this_bsz):
        # TODO Support multiple channels
        mel_idxs = source_slice[b][0]
        data[b] = pad(torch.LongTensor(mel_idxs), maxlen, PADDING_IDX)
        if args.use_metaf:
            # Here, we use the SDM as the condition.
            # |source_slice[b]| is provided as a tuple in corpus.py, with the second
            # element being the metadata dict directly from the meta.p file. So here, we 
            # load the measure_sdm field.
            # TODO We can change the key to the metadata dict to a different condition
            # from the meta.p file, but there should be an easier way to switch the
            # condition used.
            batch_conditions = source_slice[b][1]['measure_sdm']
        else:
            # TODO other similarity measures that are dynamically-calculated.
            # For instance, in the past, I used to use the hidden-state SSM from a
            # pretrained RNN as the conditions.
            # By changing the similarity measure condition, we can create an 
            # autoencoder-based model.
            pass
        conditions[b] = batch_conditions
        t = target_slice[b] 
        for j in xrange(min(args.skip_first_n_note_losses, len(t))):
            # Because we ignore padding, the first n note losses will be ignored.
            t[j] = PADDING_IDX
        target[b] = pad(torch.LongTensor(t), maxlen, PADDING_IDX)
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
    return data, conditions, target.view(-1)

def create_batch_tensor(source, batch_idx, bsz, sv):
    """ 
    Returns data/target Tensors corresponding to the batch starting at |batch_idx| 
        that's |bsz| long
    """
    def pad(tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
    start_idx = batch * bsz
    this_bsz = min(bsz, (len(source) - start_idx))
    source_slice = source[start_idx:start_idx+this_bsz]
    target_slice = [[mel[i+1] for i in range(len(mel)-1)] + [PADDING_IDX] for mel, _ in source_slice]
    maxlen = len(source_slice[0][0])
    data = torch.LongTensor(this_bsz,maxlen).zero_()
    target = torch.LongTensor(this_bsz,maxlen).zero_()
    for i in range(this_bsz):
        data[i] = pad(torch.LongTensor(source_slice[i][0]), maxlen) 
        t = target_slice[i] 
        for j in xrange(args.skip_first_n_note_losses):
            t[j] = PADDING_IDX
        target[i] = pad(torch.LongTensor(t), maxlen)
    if args.cuda: 
        data = data.cuda()
        target = target.cuda()
    return data, target.view(-1)

def batchify(source, bsz, sv):
    '''
    This function splits |source| into many |bsz|-sized Tensors. It splits 
        directly in the order that |source| is provided in, which should be
        sorted from longest to shortest.
    Must be called after the model is built (because we need 
        util.need_conditions(model, args)) 
    '''

    batch_data = {"data": [], "targets": [], "metadata": []}
    if util.need_conditions(model, args):
        batch_data["conditions"] = []
    for channel in range(len(source)):
        channel_batches = []
        channel_targets = []
        channel_metadata = []
        channel_conditions = []
        for batch_idx in range(int(len(source[channel])/bsz)):
            # For each batch, create a Tensor
            if util.need_conditions(model, args):
                data, conditions, target, = \
                    create_batch_tensor_with_conds(source[channel], batch_idx, bsz, sv)
                channel_conditions.append(conditions)
            else:
                data, target = create_batch_tensor(source[channel], batch_idx, bsz, sv) 
            channel_batches.append(data)
            channel_targets.append(target)
            if args.use_metaf:
                channel_metadata.append(get_batch_metadata(source[channel], batch_idx, bsz))
            else:
                channel_metadata.append([])
        batch_data["data"].append(channel_batches)
        batch_data["targets"].append(channel_targets)
        batch_data["metadata"].append(channel_metadata)
        if util.need_conditions(model, args):
            batch_data["conditions"].append(channel_conditions)
    if util.need_conditions(model, args):
        batch_data["conditions"] = np.asarray(batch_data["conditions"])
    return batch_data

def get_batch_variables(batches, batch, evaluation=False):
    '''
    You can't save Variables with pickle, so we wrap the Tensors for each batch in a 
        Variable each time we need to use it.
    Size of |batches|: num_channels * num_batches * num_examples_in_batch_i 
    '''
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
sv, vocabf, corpusf = load_train_vocab(args)
corpus = corpus.Corpus.load_from_corpus(sv, vocabf, corpusf, args)
print "Time elapsed", time.time() - t

args.ntokens = sv.sizes

# Load the model, either from a checkpoint, or init a new one.
if args.mode == 'train':
    if args.checkpoint != '':
        print "Loading model checkpoint"
        with open(args.checkpoint, 'rb') as f:
            model = torch.load(f)
    else:
        model = util.init_model(args)

    if args.cnn_encoder:
        cnn = cnn.CNN(args)
        if args.cuda:
            cnn.cuda()

    sigmoid = nn.Sigmoid()
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)
    params = list(model.parameters()) + list(cnn.parameters()) if args.cnn_encoder else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    print model

elif args.mode in ['get_hiddens', 'generate']:
    checkpoint = args.checkpoint if args.checkpoint != '' else util.get_savef(args, corpus)
    args.batch_size = 1
    with open(checkpoint, 'rb') as f:
        model = torch.load(f)
        model.eval()

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
        train_data = batchify(corpus.trains, args.batch_size, sv)
        valid_data = batchify(corpus.valids, args.batch_size, sv)
        test_data = batchify(corpus.tests, args.batch_size, sv)
        print "Saving train data to", f, "time elapsed", time.time() - t
        pickle.dump((train_data, valid_data, test_data), open(f, 'wb'))

    train_mb_indices = range(0, int(len(corpus.trains[0])/args.batch_size))
    valid_mb_indices = range(0, int(len(corpus.valids[0])/args.batch_size))
    test_mb_indices = range(0, int(len(corpus.tests[0])/args.batch_size))


###############################################################################
# Methods for train/RSED
###############################################################################

def get_rsed():
    ''' 
    Gets "repeating-section edit distance," which is the average distance
    between repeating sections.
    '''
    # Turn on evaluation mode which disables dropout.
    model.eval()
    path = args.path + 'train/'
    meta_dicts = util.get_meta_dicts(path, args)
    num_total_repeats = 0
    total_repeat_ed = 0
    for songf, info in random.sample(meta_dicts.items(), args.num_out):
        args.condition_piece = path + songf
        meta_dict = meta_dicts[os.path.basename(songf)]

        conditions = []
        events = gen_util.get_events(sv, args, args.condition_piece, meta_dict)
        if util.need_conditions(model, args):
            conditions = gen_util.get_conditions(sv, args, meta_dict)

        generated = generate.generate(
                model, events, conditions, meta_dict, args, corpus.vocab)
        print [e.i for e in generated[1:][:-1]]
        gen_measure_sdm = similarity.get_measure_sdm(
                [e.original for e in generated[1:][:-1]], 
                meta_dict['measure_boundaries'])
        if args.synth_data:
            repeating_measures = meta_dict['repeating_measures']
        else:
            # Get repeating sections (those with ED=0 in gold segment sdm)
            zero_idxs = np.where(meta_dict['measure_sdm'] == 0)
            repeating_measures = []
            for i in range(zero_idxs[0].shape[0]):
                if zero_idxs[0][i] <= zero_idxs[1][i]:
                    repeating_measures.append((zero_idxs[0][i], zero_idxs[1][i]))
        for repeats in repeating_measures:
            repeat_tups = list(combinations(repeats, 2))
            for i,j in repeat_tups:
                total_repeat_ed += gen_measure_sdm[i,j]
            num_total_repeats += len(repeat_tups)
    return total_repeat_ed / num_total_repeats


def evaluate(eval_data, mb_indices):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = corpus.vocab.sizes
    hidden = model.init_hidden(args.batch_size)
    for batch in mb_indices:
        data = get_batch_variables(eval_data, batch, evaluation=True)
        outputs, hidden = model(data, hidden, args)
        outputs_flat = [outputs[c].view(-1, ntokens[c]) for c in range(len(outputs))]
        total_loss += sum(
            [criterion(outputs_flat[c], data["targets"][c]) for c in range(len(outputs))]).data
        hidden = repackage_hidden(hidden)
        hidden = model.init_hidden(args.batch_size)
    return total_loss[0] / len(mb_indices)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = corpus.vocab.sizes
    hidden = model.init_hidden(args.batch_size)
    # The stochasticity in training comes from shuffling the batches, but not shuffling 
    # what is inside each batch. This is a common practice within LM training and allows
    # the examples in each batch Tensor to have similar lengths, leading to less padding
    # in each batch.
    # TODO After I wrote this file, libraries have been released for Pytorch that do 
    # automatic batching and such. It may be beneficial to move over to those libraries.
    random.shuffle(train_mb_indices)
    for batch in train_mb_indices:
        data = get_batch_variables(train_data, batch)
        hidden = repackage_hidden(hidden)
        hidden = model.init_hidden(args.batch_size)
        if args.cnn_encoder:
            if args.arch in {'readrnn'}:
                hidden['backbone'] = cnn(data["conditions"][0], args)
        outputs, hidden = model(data, hidden, args)

        word_idxs = []
        for i in range(outputs[0].size(1)):
            m, am = 0, 0
            for j in range(outputs[0].size(2)):
                if outputs[0][0,i,j].data[0] > m:
                    am = j
                    m = outputs[0][0,i,j].data[0]
            word_idxs.append(am)

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


###############################################################################
# Run the appropriate mode
###############################################################################

if args.mode == 'train':
    lr = args.lr
    best_val_loss = None
    losses = {'train': [], 'valid': []}
    train_outf = open(args.train_info_out, 'wb')
    writer = csv.writer(train_outf, delimiter=',')
    writer.writerow(['train_loss','val_loss','rsed'])

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        # Loop over epochs.
        for epoch in range(1, args.epochs+1):
            args.epoch = epoch-1
            epoch_start_time = time.time()
            train_loss = train()
            val_loss = evaluate(valid_data, valid_mb_indices)
            val_perp = math.exp(val_loss) if val_loss < 100 else float('nan')
            rsed = get_rsed()
            print('-' * 88)
            print "RSED", rsed
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} 
                    | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                        epoch, (time.time() - epoch_start_time), 
                        train_loss, val_loss, val_perp))
            print('-' * 88)
            losses["train"].append(train_loss)
            losses["valid"].append(val_loss)
            writer.writerow([train_loss,val_loss,rsed])
            # Write to file without closing
            train_outf.flush()
            os.fsync(train_outf.fileno())
            pickle.dump(losses, open(util.get_datadumpf(args, extra='curves'), 'wb'))
            # Save the model if the validation loss is the best we've seen so far.
            if best_val_loss is None or val_loss < best_val_loss:
                with open(args.save if args.save != '' else util.get_savef(args, corpus), 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the 
                # validation dataset.
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
        path = args.path + 'train/'
        if args.use_metaf:
            meta_dicts = util.get_meta_dicts(path, args)
            meta_dict = meta_dicts[os.path.basename(args.condition_piece)]
        else:
            meta_dict = None

        conditions = []
        # Note: |events| is only for conditions
        events = gen_util.get_events(sv, args, args.condition_piece, meta_dict)
        if util.need_conditions(model, args):
            conditions = gen_util.get_conditions(sv, args, meta_dict)

        generated = generate.generate(model, events, conditions, meta_dict, args, corpus.vocab, 
                end=True)
        outf = "../generated/" + args.outf + '_' + str(i) + '.mid'
        sv.events2mid([generated], outf)


elif args.mode == 'get_hiddens':
    args.epoch = 0
    get_hiddens.get_hiddens(model, args, corpus.vocab)
