import torch
from torch.autograd import Variable
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import similarity, util

def get_events(sv, args, midf, meta_dict):
    channel_event_idxs = [[] for _ in range(sv.num_channels)]
    for channel in range(sv.num_channels):
        if args.synth_data:
            origs = [sv.special_events['start'].original] + meta_dict['origs'] + [sv.special_events['end'].original]
        else:
            origs, _ = sv.mid2orig(args.condition_piece, include_measure_boundaries=args.measure_tokens, channel=channel)
        mel_idxs = [sv.orig2e[channel][o].i for o in origs]
        for idx in mel_idxs:
            channel_event_idxs[channel].append(idx)
    return channel_event_idxs

def get_conditions(sv, args, meta_dict):
    channel_conditions = [[] for _ in range(sv.num_channels)]
    for channel in range(sv.num_channels):
        if args.use_metaf:
            ssm = meta_dict['measure_sdm']
        else:
            # TODO See equivalent comment in main.py
            pass
    channel_conditions[channel] = [ssm] 
    conditions = [channel_conditions[c] for c in range(sv.num_channels)]
    return conditions

def make_data_dict(args, sv, need_conditions):
    ''' 
    Returns a tuple. 
    Both outputs are lists of lists, one sublist for each channel
    '''
    data = {}
    data["data"] = [Variable(torch.FloatTensor(1, 1).zero_().long() + sv.special_events["start"].i, volatile=True)]
    if need_conditions:
        data["conditions"] = [Variable(torch.LongTensor(1, 1).zero_(), volatile=True) for c in range(sv.num_channels)] 
    if args.cuda:
        for c in range(sv.num_channels):
            data["data"][c].data = data["data"][c].data.cuda()
            if need_conditions:
                data["conditions"][c].data = data["conditions"][c].data.cuda()
    return data

