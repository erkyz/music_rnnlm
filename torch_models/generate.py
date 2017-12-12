import torch
from torch.autograd import Variable

import sys, os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data, util

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
parser.add_argument('--condition_measures', type=int, default=0,
                    help='number of bars to condition the generation on')
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

def get_events(condition_piece, cuda, sv):
    ''' 
    Returns a tuple. 
    Both outputs are lists of lists, one sublist for each channel
    '''
    channel_events = [[] for _ in range(sv.num_channels)]
    if args.condition_piece != "":
        for channel in range(sv.num_channels):
            curr_measure = 0
            origs, _ = sv.mid2orig(args.condition_piece, channel)
            for orig in origs:
                event = sv.orig2e[channel][orig]
                channel_events[channel].append(event)
                if event == sv.special_events["measure"]:
                    curr_measure += 1
                if curr_measure == args.condition_measures:
                    break
        channel_event_idxs = [[e.i for e in channel_events[c]] for c in range(sv.num_channels)]
        input = [Variable(torch.LongTensor(channel_event_idxs[c]).view(1,-1), volatile=True) for c in range(sv.num_channels)]
    else:
        # Input should be START token
        # TODO
        input = [Variable(torch.FloatTensor(1, 1).zero_().long() + sv.special_events["start"].i, volatile=True)]
    if args.cuda:
        for c in range(sv.num_channels):
            input[c].data = input[c].data.cuda()
    return input, channel_events

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

for i in range(args.num_out):
    hidden = model.init_hidden(1) # batch size of 1
    torch.manual_seed(i)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(i)
    gen_input, generated_events = get_events(args.condition_piece, args.cuda, sv)
    measure_progress = 0 
    while len(generated_events) < args.max_events:
        if args.arch == "hrnn":       
            outputs, hidden = model(gen_input, hidden, sv.special_events['measure'].i)
        else:
            outputs, hidden = model(gen_input, hidden)
        outputs = [outputs[c][0][-1].view((1,1,-1)) for c in range(sv.num_channels)]
        word_weights = [outputs[c].squeeze().data.div(args.temperature).exp().cpu() for c in range(sv.num_channels)]
        word_idxs = [torch.multinomial(word_weights[c], 1)[0] for c in range(sv.num_channels)]
        '''
        if args.progress_tokens:
            measure_progress += word_idxs[sv.duration_channel]
        '''
        for c in range(sv.num_channels):
            '''
            if args.progress_tokens and c == sv.measure_channel:
                gen_input[c].data.fill_(sv.orig2e[sv.measure_channel][measure_progress].i)
            else:
            '''
            gen_input[c].data.fill_(word_idxs[c])
            generated_events[c].append(sv.i2e[c][word_idxs[c]])
        if word_idxs[0] == sv.special_events["end"].i or word_idxs[0] == sv.special_events["end"].i:
            break
    print len(generated_events[0])
    sv.events2mid(generated_events, "../../generated/" + args.outf + str(i) + '.mid')


