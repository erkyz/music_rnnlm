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
    hidden = model.init_hidden(1)
    if args.arch == 'prnn':
        prevs = [hidden["parallel"]]
    elif args.arch == 'mrnn':
        prevs = [[]] # TODO 
    elif args.arch == 'ernn':
        prevs = [] 
    else: # XRNN
        prevs = [hidden]
    gen_data = gen_util.make_data_dict(args, sv)
    gen_data["cuda"] = args.cuda
    generated_events = [[sv.i2e[c][events[c][0]]] for c in range(sv.num_channels)]
    args.epoch = 0
    # Always start the generation with START
    word_idxs = [events[c][0] for c in range(sv.num_channels)] 
   
    # Fill the entire conditions Tensor first, because we'll need to see all of it
    # in the RNN.
    for t in range(min(args.max_events, len(events[0]))):
        if args.arch in util.CONDITIONALS:
            for c in range(sv.num_channels):
                gen_data["conditions"][c] = torch.cat(
                        [gen_data["conditions"][c],
                        (torch.cuda.LongTensor(1,1).zero_() + conditions[c][t] 
                        if args.cuda else
                        torch.LongTensor(1,1).zero_() + conditions[c][t])])
                if t == 0:
                    gen_data["conditions"][c] = gen_data["conditions"][c][1:]
    # We want to emulate (bsz,seqlen) even though bsz=1 
    gen_data["conditions"][0] = gen_data["conditions"][0].permute(1,0)
    if args.use_metaf:
        gen_data["metadata"] = [[meta_dict["segments"]]]

    for t in range(min(args.max_events, len(events[0]))):
        for c in range(sv.num_channels):
            if t < args.condition_notes:
                gen_data["data"][c].data.fill_(events[c][t])
            else:
                gen_data["data"][c].data.fill_(word_idxs[c])

        if args.arch == "hrnn":
            outputs, hidden = model(gen_data, hidden, sv.special_events['measure'].i)
        elif args.arch in util.CONDITIONALS:
            # prevs modified in place
            outputs, hidden = model(gen_data, hidden, args, prevs, t)
        else:
            outputs, hidden = model(gen_data, hidden, args)
        

        # print outputs
        word_weights = [F.softmax(outputs[c].squeeze().data.div(args.temperature)).cpu() for c in range(sv.num_channels)]
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


