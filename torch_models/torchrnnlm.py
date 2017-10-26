

import torch.nn as nn
from torch.autograd import Variable
import time, util

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.1, attention=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # Choose attention model
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
	print input
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)


class HRNNModel(nn.Module):
    """Hierarchical LSTM"""
    '''MUST take batch size of one'''

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.1):
        super(RNNModel, self).__init__()
        low_out_dim = 30
        # TODO different hyperparams
        self.low = RNNModel(rnn_type, low_out_dim, ninp, nhid, nlayers, dropout)
        self.high = RNNModel(rnn_type, ntoken, low_out_dim, nhid, nlayers, dropout)
        self.init_weights()

    def init_weights(self):
        self.low.init_weights()
        self.high.init_weights()

    # TODO really shouldn't be recursive...
    # Dude this is a computation graph, everything needs to be an op.
    def forward_recurse(self, input, hidden_low, hidden_high):
        print input[0]
        if input[0][0].value == 0:
            hidden_low = self.high(hidden_low, hidden_high)
        step_inp = input.narrow(dimension=1, start=1, 1) 
        step_out, hidden_low = self.low(step_inp, hidden)
        next_inp = input.narrow(dimension=1, start=1, input.size()[1]-1) 
        return torch.cat(step_out, self.forward_step(next_inp, hidden_low))

    def forward(self, input, hidden_low, hidden_high):
        emb = self.drop(self.encoder(input))
        output = self.forward_recurse(emb, hidden_low, hidden_high)
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1), 
                hidden_low, hidden_high)

    def init_hidden(self, bsz):
        return self.low.init_hidden(bsz), self.high.init_hidden(bsz)

