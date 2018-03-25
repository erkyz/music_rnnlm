from __future__ import division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

import sys, os
import argparse, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data, util, similarity, beam_search, gen_util

'''
parser = argparse.ArgumentParser()

# Data stuff
parser.add_argument('--data', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--tmp_prefix', type=str, default="../tmp/cmaj_nott",
                    help='tmp directory + prefix for tmp files')
parser.add_argument('--checkpoint', type=str, default='../tmp/model.pt',
                    help='model checkpoint to use')

# RNN params
parser.add_argument('--arch', type=str, default='base')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
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
parser.add_argument('--vanilla_ckpt', type=str,  default='',
                    help='pretrained vanilla model dir')
parser.add_argument('--window', type=int, default=8,
                    help='window size for note-based moving edit distances')
parser.add_argument('--c', type=float, default=2,
                    help='number of measures to base the note-based ED window off of')
parser.add_argument('--distance_threshold', type=int, default=3,
                    help='distance where below, we consider windows sufficiently similar')
parser.add_argument('--most_recent', action='store_true',
                    help='whether we repeat the most recent similar or earliest similar')

# Meta-training stuff
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
args = parser.parse_args()
print args

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

'''

def get_hiddens(model, h0, args, sv):
    prev_hs = [h0]
    gen_data = gen_util.make_data_dict(args, sv)
    gen_data["cuda"] = args.cuda
    if args.arch in util.CONDITIONALS:
        events, conditions = gen_util.get_events_and_conditions(sv, args, vanilla_model)
    else:
        events = gen_util.get_events(sv, args, args.condition_piece)
    print zip(events[0], range(len(events[0])))
    # print zip(conditions[0], range(len(conditions[0])))

    hidden = h0
    for t in range(min(args.max_events, len(events[0]))):
        if args.arch in util.CONDITIONALS:
            for c in range(sv.num_channels):
                gen_data["conditions"][c].data.fill_(conditions[c][t])

        for c in range(sv.num_channels):
            gen_data["data"][c].data.fill_(events[c][t])

        if args.arch == "hrnn":
            outputs, hidden = model(gen_data, hidden, sv.special_events['measure'].i)
        elif args.arch == 'vine' or args.arch == 'xrnn':
            # prev_hs modified in place
            outputs, hidden = model(gen_data, hidden, prev_hs)
        else:
            outputs, hidden = model(gen_data, hidden)
            prev_hs.append(hidden)

    print len(prev_hs)
    print prev_hs[0]
    sims = similarity.get_hid_sim(prev_hs, args, False)
    print sims
    pickle.dump(sims, open("../tmp/test1.p", 'wb'))
    sims = similarity.get_hid_sim(prev_hs, args, True)
    pickle.dump(sims, open("../tmp/test2.p", 'wb'))

    '''
    plt.imshow(sims, cmap='gray', interpolation='nearest')
    plt.savefig('test.png')
    '''

