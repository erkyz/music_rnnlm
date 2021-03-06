import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys, os
import argparse

import corpus, similarity, gen_util, util


NO_INFO_EVENT_IDX = 3

def generate(model, events, conditions, meta_dict, args, sv, end=False):
    model.eval()
    bsz = 1
    hidden = model.init_hidden(bsz)
    if args.arch == 'attn':
        prev_data = []
    elif args.arch == 'readrnn':
        prev_data = {'delta_tilde': [None for b in range(bsz)], 'encs': [[] for b in range(bsz)]}
    else: 
        prev_data = None # not needed
    gen_data = gen_util.make_data_dict(args, sv, util.need_conditions(model, args))
    gen_data["conditions"] = conditions
    gen_data["cuda"] = args.cuda
    generated_events = [[sv.i2e[c][events[c][0]]] for c in range(sv.num_channels)]
    args.epoch = 0
    # Always start the generation with START
    word_idxs = [events[c][0] for c in range(sv.num_channels)] 

    # Pass in the entirety of conditions even though we only look at up to t
    if args.use_metaf:
        gen_data["metadata"] = [[meta_dict["measure_boundaries"]]]

    for t in range(min(args.max_events, len(events[0]))):
        for c in range(sv.num_channels):
            if t < args.condition_notes:
                gen_data["data"][c].data.fill_(events[c][t])
            else:
                gen_data["data"][c].data.fill_(word_idxs[c])

        if util.need_conditions(model, args):
            # prev_data modified in place
            outputs_t, hidden = model(gen_data, hidden, args, prev_data, t)
        else:
            outputs_t, hidden = model(gen_data, hidden, args)
        
        word_weights = [F.softmax(outputs_t[c].squeeze().data.div(args.temperature)).cpu() for c in range(sv.num_channels)]
        word_idxs = [torch.multinomial(word_weights[c], 1)[0].data[0] for c in range(sv.num_channels)] 
        for c in range(sv.num_channels):
            if t < args.condition_notes:
                # Always include START as the first generated token
                generated_events[c].append(sv.i2e[c][events[c][t]])
            else:
                generated_events[c].append(sv.i2e[c][word_idxs[c]])

        if word_idxs[0] == sv.special_events["end"].i and end:
            break

    return generated_events[0]

