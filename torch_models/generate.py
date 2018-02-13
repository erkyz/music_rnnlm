import torch
from torch.autograd import Variable
import torch.nn.functional as F

import sys, os
import argparse
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
parser.add_argument('--max_events', type=int, default='100', # TODO
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
    data["data"] = [Variable(torch.FloatTensor(args.beam_size, 1).zero_().long() + sv.special_events["start"].i, volatile=True)]
    if args.arch == 'crnn':
        data["conditions"] = [Variable(torch.LongTensor(args.beam_size, 1).zero_(), volatile=True) for c in range(sv.num_channels)] 
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
        channel_conditions[channel] = similarity.get_future_from_past(origs[1:], args)

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

hidden = model.init_hidden(args.beam_size) 
gen_data = make_data_dict()
events, conditions = get_events_and_conditions(sv)

beam = beam_search.Beam(args.beam_size, sv, cuda=True)

# word_idxs = None # TODO

for t in range(args.max_events):
    print ""
    if beam.done: 
        break

    # Set the proper condition for this step
    for c in range(sv.num_channels):
        if args.arch == 'crnn':
            if t < len(conditions[c]) or t == 0:
                gen_data["conditions"][c].data.fill_(conditions[c][t]) # TODO
            else:
                gen_data["conditions"][c].data.fill_(0) # TODO # TODO 

    for c in range(sv.num_channels):
        if t < args.condition_notes:
            gen_data["data"][c].data.fill_(events[c][t])
        else:
            for i in range(args.beam_size):
                # gen_data["data"][c][i] = word_idxs[c][0]
                gen_data["data"][c][i] = beam.get_current_state(c)[i]

    if args.arch == "hrnn":
        outputs, hidden = model(gen_data, hidden, sv.special_events['measure'].i)
    else:
        outputs, hidden = model(gen_data, hidden)

    word_weights = [F.softmax(outputs[c].squeeze(1).data.div(args.temperature)).contiguous() for c in range(sv.num_channels)]
    word_idxs = [torch.multinomial(word_weights[c], 1)[0] for c in range(sv.num_channels)] # TODO option to sample.

    # Starting from the last note in the notes we condition on, we should advance the beam.
    # Before that, we're just conditioning.
    if t >= args.condition_notes-1 and beam.advance(word_weights):
        # If the beam search is complete, exit.
        break
    
    ''' 
    for c in range(sv.num_channels):
        gen_data["data"][c].data.fill_(word_idxs[c])
        if args.arch == 'crnn':
            i = len(generated_events[c])
            if i < len(conditions[c]):
                gen_data["conditions"][c].data.fill_(conditions[c][i])
            else:
                gen_data["conditions"][c].data.fill_(0)
        generated_events[c].append(sv.i2e[c][word_idxs[c]])
    if word_idxs[0] == sv.special_events["end"].i or word_idxs[0] == sv.special_events["end"].i:
        break
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

