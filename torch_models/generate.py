import torch
from torch.autograd import Variable

import sys, os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data, util

parser = argparse.ArgumentParser()

# Model parameters.
parser.add_argument('--data', type=str, default='../music_data/Nottingham/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='../tmp/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='test.mid',
                    help='output file for generated text')
parser.add_argument('--max_events', type=int, default='200',
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
    events = []
    if args.condition_piece != "":
        with open(args.condition_piece) as f:
            tuples = util.mid2tuples(f)
        curr_measure = 0
        for tup in tuples:
            event = sv[tup]
            events.append(event.i)
            if event == sv.special_events["measure"]:
                curr_measure += 1
            if curr_measure == args.condition_measures:
                break
        input = Variable(torch.LongTensor(events).view(1,-1), volatile=True)
    else:
        # Input should be START token
        input = Variable(torch.FloatTensor(1, 1).zero_().long() + sv.special_events["start"].i, volatile=True)
    if args.cuda:
        input.data = input.data.cuda()
    return input, events


sv = util.SimpleVocab.load_from_corpus(args.data, "../tmp/nott_sv.p")
ntokens = len(sv)

for i in range(args.num_out):
    hidden = model.init_hidden(1) # batch size of 1
    torch.manual_seed(i)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(i)
    gen_input, generated_events = get_events(args.condition_piece, args.cuda, sv)
    while len(generated_events) < args.max_events:
        output, hidden = model(gen_input, hidden)
        output = output[0][-1].view((1,1,-1))
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        gen_input.data.fill_(word_idx)
        curr = sv[word_idx]
        generated_events.append(curr)
        if curr == sv.special_events["end"].i: break
    sv.list2mid(generated_events, "../../generated/" + args.outf + str(i) + '.mid')


