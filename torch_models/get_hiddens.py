from __future__ import division

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import sys, os
import argparse, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data, util, similarity, beam_search

parser = argparse.ArgumentParser()

# Model parameters.
parser.add_argument('--data', type=str, default='../music_data/CMaj_Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--vocabf', type=str, default="../tmp/cmaj_nott_sv.p",
                    help='location of the saved vocabulary')
parser.add_argument('--corpusf', type=str, default="../tmp/cmaj_nott_corpus.p",
                    help='location of the saved corpus, or where to save it')
parser.add_argument('--checkpoint', type=str, default='../tmp/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='test.mid',
                    help='output file for generated text')
parser.add_argument('--max_events', type=int, default='500',
                    help='number of words to generate')
parser.add_argument('--num_out', type=int, default='10',
                    help='number of melodies to generate')
parser.add_argument('--beam_size', type=int, default='3',
                    help='beam size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--condition_piece', type=str, default="",
                    help='midi piece to condition on')
parser.add_argument('--condition_notes', type=int, default=0,
                    help='number of notes to condition the generation on')
parser.add_argument('--window', type=int, default=8,
                    help='window size for note-based moving edit distances')
parser.add_argument('--distance_threshold', type=int, default=3,
                    help='distance where below, we consider windows sufficiently similar')
parser.add_argument('--arch', type=str, default='base')
parser.add_argument('--progress_tokens', action='store_true',
                    help='whether to condition on the amount of time left until the end \
                            of the measure')
parser.add_argument('--factorize', action='store_true',
                    help='whether to factorize embeddings')

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

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
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
    if args.arch == 'crnn':
        data["conditions"] = [Variable(torch.LongTensor(1, 1).zero_(), volatile=True) for c in range(sv.num_channels)] 
    if args.cuda:
        for c in range(sv.num_channels):
            data["data"][c].data = data["data"][c].data.cuda()
            if args.arch == 'crnn':
                data["conditions"][c].data = data["conditions"][c].data.cuda()
    return data


def get_events_and_conditions(sv):
    if args.condition_piece == "":
        return [], []

    channel_events = [[] for _ in range(sv.num_channels)]
    channel_conditions = [[] for _ in range(sv.num_channels)]

    for channel in range(sv.num_channels):
        curr_note = 0
        origs, _ = sv.mid2orig(args.condition_piece, channel)
        channel_conditions[channel] = similarity.get_min_past_distance(origs[1:], args)

        for orig in origs:
            if orig[0] == "rest":
                continue # TODO 
            event = sv.orig2e[channel][orig]
            channel_events[channel].append(event)
            curr_note += 1
            if curr_note == args.condition_notes:
                break
    channel_event_idxs = [[e.i for e in channel_events[c]] for c in range(sv.num_channels)]
    conditions = [channel_conditions[c] for c in range(sv.num_channels)]
    return channel_event_idxs, conditions


def get_hid_sim(hiddens):
    sims = np.zeros([len(hiddens), len(hiddens)])
    for i in range(len(hiddens)):
        for j in range(i, len(hiddens)):
            l = hiddens[0][i] if args.arch == 'LSTM' else hiddens[i]
            r = hiddens[0][j] if args.arch == 'LSTM' else hiddens[j]
            # cosine similarity
            sims[i,j] = sims[j,i] = torch.dot(l,r) / (torch.norm(l) * torch.norm(r))

if args.factorize:
    if args.progress_tokens:
        vocabf = args.vocabf + '_factorized_measuretokens.p'
        corpusf = args.corpusf + '_factorized_measuretokens.p'
        print vocabf
        sv = util.FactorPDMVocab.load_from_corpus(args.data, vocabf)
    else:
        vocabf = args.vocabf + '_factorized.p'
        corpusf = args.corpusf + '_factorized.p'
        sv = util.FactorPitchDurationVocab.load_from_corpus(args.data, vocabf)
else:
    vocabf = args.vocabf + '.p'
    corpusf = args.corpusf + '.p'
    sv = util.PitchDurationVocab.load_from_corpus(args.data, vocabf)

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
    gen_data = make_data_dict()
    events, conditions = get_events_and_conditions(sv)

    hiddens = []

    for t in range(len(events[0])):
        for c in range(sv.num_channels):
            fill_val = NO_INFO_EVENT_IDX
            '''
            if args.arch == 'crnn' and t < len(conditions[c]):
                prev_idx = conditions[c][t]
                # TODO make this work for CRNN.
                if prev_idx != -1 and prev_idx < len(generated_events[c])-1:
                    fill_val = similarity.diff((generated_events[c][prev_idx].original,generated_events[c][prev_idx+1].original))[0]+1
            gen_data["conditions"][c].data.fill_(fill_val)
            '''

        for c in range(sv.num_channels):
            gen_data["data"][c].data.fill_(events[c][t])

        if args.arch == "hrnn":
            outputs, hidden = model(gen_data, hidden, sv.special_events['measure'].i)
        else:
            outputs, hidden = model(gen_data, hidden)
        hiddens.append(hidden)

    print hiddens
    sims = get_hid_sim(hiddens)
    print sims
    pickle.dump(hiddens, open("../tmp/test1.p", 'wb'))
    pickle.dump(sims, open("../tmp/test2.p", 'wb'))


