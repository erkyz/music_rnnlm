import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import sys, os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data, util, similarity, beam_search, gen_util


NO_INFO_EVENT_IDX = 3

def generate(model, events, conditions, meta_dict, args, sv, vanilla_model=None, end=False):
    model.eval()
    bsz = 1
    hidden = model.init_hidden(bsz)
    if args.arch == 'prnn':
        prev_data = [hidden["parallel"]]
    elif args.arch == 'attn':
        prev_data = []
    elif args.arch == 'mrnn':
        prev_data = {'score_softmax': [None for b in range(bsz)], 'encs': [[] for b in range(bsz)]}
    elif args.arch == 'ernn':
        prev_data = [] 
    else: # XRNN
        prev_data = [hidden]
    gen_data = gen_util.make_data_dict(args, sv)
    gen_data["conditions"] = conditions
    gen_data["cuda"] = args.cuda
    generated_events = [[sv.i2e[c][events[c][0]]] for c in range(sv.num_channels)]
    args.epoch = 0
    # Always start the generation with START
    word_idxs = [events[c][0] for c in range(sv.num_channels)] 

    # Pass in the entirety of conditions even though we only look at up to t
    if args.use_metaf:
        gen_data["metadata"] = [[meta_dict["segments"]]]

    for t in range(min(args.max_events, len(events[0]))):
        for c in range(sv.num_channels):
            if t < args.condition_notes:
                gen_data["data"][c].data.fill_(events[c][t])
            else:
                gen_data["data"][c].data.fill_(word_idxs[c])

        if args.arch == "hrnn":
            outputs_t, hidden = model(gen_data, hidden, sv.special_events['measure'].i)
        elif args.conditional_model:
            # prev_data modified in place
            outputs_t, hidden = model(gen_data, hidden, args, prev_data, t)
        else:
            outputs_t, hidden = model(gen_data, hidden, args)
        

        # print outputs_t
        word_weights = [F.softmax(outputs_t[c].squeeze().data.div(args.temperature)).cpu() for c in range(sv.num_channels)]
        # print word_weights
        word_idxs = [torch.multinomial(word_weights[c], 1)[0].data[0] for c in range(sv.num_channels)] 
        # print word_idxs
        # print ""
        for c in range(sv.num_channels):
            if t < args.condition_notes:
                # Always include START as the first generated token
                generated_events[c].append(sv.i2e[c][events[c][t]])
            else:
                generated_events[c].append(sv.i2e[c][word_idxs[c]])

        if word_idxs[0] == sv.special_events["end"].i and end:
            break


    return generated_events[0]


