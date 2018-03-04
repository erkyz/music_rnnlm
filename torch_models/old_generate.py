import torch
from torch.autograd import Variable
import torch.nn.functional as F

import sys, os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data, util, similarity, beam_search

parser = argparse.ArgumentParser()

# Data stuff
parser.add_argument('--data', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--tmp_prefix', type=str, default="../tmp/cmaj_nott",
                    help='tmp directory + prefix for tmp files')
parser.add_argument('--checkpoint', type=str, default='../tmp/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='test.mid',
                    help='output file for generated text')

# Generate stuff
parser.add_argument('--max_events', type=int, default='250',
                    help='number of words to generate')
parser.add_argument('--num_out', type=int, default='10',
                    help='number of melodies to generate')
parser.add_argument('--beam_size', type=int, default='3',
                    help='beam size')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--condition_piece', type=str, default="",
                    help='midi piece to condition on')
parser.add_argument('--condition_notes', type=int, default=0,
                    help='number of notes to condition the generation on')

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

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
print model
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

def make_data_dict():
    ''' 
    Returns a tuple. 
    Both outputs are lists of lists, one sublist for each channel
    '''
    data = {}
    data["data"] = [Variable(torch.FloatTensor(1, 1).zero_().long() + sv.special_events["start"].i, volatile=True)]
    data["conditions"] = [Variable(torch.LongTensor(1, 1).zero_(), volatile=True) for c in range(sv.num_channels)] 
    if args.cuda:
        for c in range(sv.num_channels):
            data["data"][c].data = data["data"][c].data.cuda()
            data["conditions"][c].data = data["conditions"][c].data.cuda()
    return data


def get_events_and_conditions(sv):
    if args.condition_piece == "":
        return [], []

    channel_events = [[] for _ in range(sv.num_channels)]
    channel_conditions = [[] for _ in range(sv.num_channels)]

    for channel in range(sv.num_channels):
        curr_note = 0
        origs, _ = sv.mid2orig(args.condition_piece, include_measure_boundaries=args.measure_tokens)
        melody2, _ = sv.mid2orig(args.condition_piece, include_measure_boundaries=True)
        args.window = max(int(args.c*similarity.get_avg_dist_between_measures(melody2, sv)), similarity.MIN_WINDOW)
        print "window", args.window
        channel_conditions[channel] = similarity.get_prev_match_idx(origs[1:], args, sv)
        for orig in origs:
            if orig[0] == "rest":
                event = sv.orig2e[channel][("rest", orig[1])]
            else:
                event = sv.orig2e[channel][orig]
            channel_events[channel].append(event)
            curr_note += 1
    channel_event_idxs = [[e.i for e in channel_events[c]] for c in range(sv.num_channels)]
    conditions = [channel_conditions[c] for c in range(sv.num_channels)]
    return channel_event_idxs, conditions

sv, _, _ = util.load_train_vocab(args)

NO_INFO_EVENT_IDX = 3

for i in range(args.num_out):
    print ""
    torch.manual_seed(i)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(i) 

    hidden = model.init_hidden(1) 
    prev_hs = [hidden]
    gen_data = make_data_dict()
    gen_data["cuda"] = args.cuda
    events, conditions = get_events_and_conditions(sv)
    generated_events = [[] * sv.num_channels] # TODO

    for t in range(min(args.max_events, len(conditions[0]))):
        for c in range(sv.num_channels):
            '''
            fill_val = NO_INFO_EVENT_IDX
            if args.arch in util.CONDITIONALS and t < len(conditions[c]):
                prev_idx = conditions[c][t]
                if prev_idx != -1 and prev_idx < len(generated_events[c])-1:
                    fill_val = similarity.diff((generated_events[c][prev_idx].original,generated_events[c][prev_idx+1].original))[0]+1
            '''
            gen_data["conditions"][c].data.fill_(conditions[c][t])

        for c in range(sv.num_channels):
            if t < args.condition_notes:
                gen_data["data"][c].data.fill_(events[c][t])
            else:
                gen_data["data"][c].data.fill_(word_idxs[c])

        if args.arch == "hrnn":
            outputs, hidden = model(gen_data, hidden, sv.special_events['measure'].i)
        elif args.arch == 'vine' or args.arch == 'xrnn':
            # prev_hs modified in place
            outputs, hidden = model(gen_data, hidden, prev_hs)
        else:
            outputs, hidden = model(gen_data, hidden)

        word_weights = [F.softmax(outputs[c].squeeze().data.div(args.temperature)).cpu() for c in range(sv.num_channels)]
        word_idxs = [torch.multinomial(word_weights[c])[0].data[0] for c in range(sv.num_channels)] 
        for c in range(sv.num_channels):
            if t < args.condition_notes+1:
                generated_events[c].append(sv.i2e[c][events[c][t]])
            else:
                generated_events[c].append(sv.i2e[c][word_idxs[c]])

        if t >= args.condition_notes and word_idxs[0] == sv.special_events["end"].i:
            break

    print i, zip(
            [x.i for x in generated_events[0]], conditions[0], range(len(generated_events[0])))
    sv.events2mid([generated_events[0]], "../../generated/" + args.outf + '_' + str(i) + '.mid')

'''
allHyp, allScores = [], []
n_best = max(args.beam_size, 1)
scores, ks = beam.sort_best()
# allScores += [scores[:n_best]]

for k in range(n_best):
    hyp = beam.get_hyp(k)
    events = [[sv.i2e[c][hyp[c][i]] for i in range(len(hyp[c]))] for c in range(len(hyp))]
    print [[hyp[c][i] for i in range(len(hyp[c]))] for c in range(len(hyp))]
    sv.events2mid(events, "../../generated/" + args.outf + str(k) + '.mid')
'''

