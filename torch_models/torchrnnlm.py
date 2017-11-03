
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, util

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.1, attention=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
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
        emb = self.drop(self.encoder(torch.t(input)))
        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.drop(torch.transpose(output, 0, 1))

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

class RNNCellModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    """One layer only."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.1, attention=False):
        super(RNNCellModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type + "Cell")(ninp, nhid, nlayers)
            print self.rnn
        '''
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        '''
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
        output = []
        h_t, c_t = hidden
        emb = self.drop(self.encoder(input))
        for i, emb_t in enumerate(emb.chunk(emb.size(1), dim=1)):
            h_t, c_t = self.rnn(emb_t, (h_t, c_t))
            output += [h_t]
        output = torch.stack(output, 1).squeeze(2)
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), (h_t, c_t)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())



class AttentionRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    """One layer only."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.1, attention=False):
        super(AttentionRNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type + "Cell")(ninp, nhid, nlayers)
            print self.rnn
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
        output = []
        h_t, c_t = hidden
        emb = self.drop(self.encoder(input))
        for i, emb_t in enumerate(emb.chunk(emb.size(1), dim=1)):
            h_t, c_t = self.rnn(emb_t, (h_t, c_t))
            output += [h_t]
        output = torch.stack(output, 1).squeeze(2)
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), (h_t, c_t)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())


# TODO pass vocab
MEASURE_TOKEN_IDX = 3


class HRNNModel(nn.Module):
    """Hierarchical LSTM"""
    # TODO different hyperparams (esp learning rate), type of model

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers=1, dropout=0.1):
        super(HRNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.low = getattr(nn, rnn_type+'Cell')(ninp+nhid, nhid, nlayers)
            self.high = getattr(nn, rnn_type+'Cell')(ninp, nhid, nlayers)
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

    def forward(self, input, hiddens):
        output = []
        hidden_low, hidden_high = hiddens
        h_low_t, c_low_t = hidden_low
        h_high_t, c_high_t = hidden_high
        for i, x_t in enumerate(input.chunk(input.size(1), dim=1)):
            # TODO make these faster.
            filter_t = Variable((x_t.data == MEASURE_TOKEN_IDX).expand(
                x_t.size(0), h_high_t.size(1)).type(torch.cuda.FloatTensor))
            emb_t = self.drop(self.encoder(x_t))
            inp_t = torch.cat([emb_t, h_high_t.view(x_t.size(0),1,-1)], 2)
            h_low_t, c_low_t = self.low(inp_t, (h_low_t, c_low_t))
            h_high_tmp, c_high_tmp = self.high(h_low_t, (h_high_t, c_high_t))
            h_high_t = torch.mul(filter_t, h_high_tmp) + torch.mul(1-filter_t, h_high_t)
            c_high_t = torch.mul(filter_t, c_high_tmp) + torch.mul(1-filter_t, c_high_t)
            output += [h_low_t]
        output = torch.stack(output, 1).squeeze(2)
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), \
            ((h_low_t, c_low_t), (h_high_t, c_high_t))

    
    # returns tuple for hidden states of low and high LSTMs
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return ((Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_())),
                    (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_())))


