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
import corpus, util, similarity, gen_util


def get_hiddens(model, args, sv):
    prev_hs = [model.init_hidden(1)]
    gen_data = gen_util.make_data_dict(args, sv)
    gen_data["cuda"] = args.cuda
    if model.need_conditions:
        events, conditions = gen_util.get_events_and_conditions(sv, args)
    else:
        events = gen_util.get_events(sv, args, args.condition_piece)

    for t in range(min(args.max_events, len(events[0]))):
        if model.need_conditions:
            for c in range(sv.num_channels):
                gen_data["conditions"][c].data.fill_(conditions[c][t])

        for c in range(sv.num_channels):
            gen_data["data"][c].data.fill_(events[c][t])

        if args.arch == "hrnn":
            outputs, hidden = model(gen_data, prev_hs[-1], sv.special_events['measure'].i)
        elif args.arch == 'vine' or args.arch == 'xrnn':
            # prev_hs modified in place
            outputs, hidden = model(gen_data, prev_hs[-1], args, prev_hs)
        else:
            outputs, hidden = model(gen_data, prev_hs[-1], args)
            prev_hs.append(hidden)

    sims = similarity.get_hid_sim(prev_hs, args, False)
    print sims
    pickle.dump(sims, open("../tmp/test1.p", 'wb'))
    sims = similarity.get_hid_sim(prev_hs, args, True)
    pickle.dump(sims, open("../tmp/test2.p", 'wb'))

    '''
    plt.imshow(sims, cmap='gray', interpolation='nearest')
    plt.savefig('test.png')
    '''

