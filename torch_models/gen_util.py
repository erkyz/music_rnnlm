import torch
from torch.autograd import Variable
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import similarity, util

def get_events(sv, args, midf):
    channel_event_idxs = [[] for _ in range(sv.num_channels)]
    for channel in range(sv.num_channels):
        origs, _ = sv.mid2orig(args.condition_piece, include_measure_boundaries=args.measure_tokens, channel=channel) if not args.use_metaf else ([sv.special_events['start']] + meta_dict['origs'] + [sv.special_events['end']], 0)
        mel_idxs = [sv.orig2e[channel][o].i for o in origs]
        for idx in mel_idxs:
            channel_event_idxs[channel].append(idx)
    return channel_event_idxs

def get_events_and_conditions(sv, args, vanilla_model, meta_dict):
    if args.condition_piece == "":
        return [], []

    channel_events = [[] for _ in range(sv.num_channels)]
    channel_conditions = [[] for _ in range(sv.num_channels)]

    for channel in range(sv.num_channels):
        origs, _ = sv.mid2orig(args.condition_piece, include_measure_boundaries=args.measure_tokens, channel=channel) if not args.use_metaf else ([sv.special_events['start'].original] + meta_dict['origs'] + [sv.special_events['end'].original], 0)
        if args.use_metaf:
            ssm = meta_dict['ssm']
        elif vanilla_model is None:
            melody2, _ = sv.mid2orig(args.condition_piece, include_measure_boundaries=True, channel=channel)
            args.window = max(int(args.c*similarity.get_avg_dist_between_measures(melody2, sv)), similarity.MIN_WINDOW)
            ssm = similarity.get_note_ssm_future(origs[1:], args, bnw=True)[0]
            # TODO ssm is the wrong length here. 
        else:
            events = get_events(sv, args, args.condition_piece)
            ssm = similarity.get_rnn_ssm(args, sv, vanilla_model, events)
        channel_conditions[channel] = similarity.get_prev_match_idx(ssm, args, sv)
        for orig in origs:
            if orig[0] == "rest":
                event = sv.orig2e[channel][("rest", orig[1])]
            else:
                event = sv.orig2e[channel][orig]
            channel_events[channel].append(event)
    channel_event_idxs = [[e.i for e in channel_events[c]] for c in range(sv.num_channels)]
    conditions = [channel_conditions[c] for c in range(sv.num_channels)]
    return channel_event_idxs, conditions

def make_data_dict(args, sv):
    ''' 
    Returns a tuple. 
    Both outputs are lists of lists, one sublist for each channel
    '''
    data = {}
    data["data"] = [Variable(torch.FloatTensor(1, 1).zero_().long() + sv.special_events["start"].i, volatile=True)]
    if args.arch in util.CONDITIONALS:
        data["conditions"] = [Variable(torch.LongTensor(1, 1).zero_(), volatile=True) for c in range(sv.num_channels)] 
    if args.cuda:
        for c in range(sv.num_channels):
            data["data"][c].data = data["data"][c].data.cuda()
            if args.arch in util.CONDITIONALS:
                data["conditions"][c].data = data["conditions"][c].data.cuda()
    return data

