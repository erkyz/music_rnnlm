
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, util

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.1, attention=False):
	"""
	ninp = size of input embedding
	"""
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


class RNNCellModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.1, attention=False):
	"""
	ninp = size of input embedding
	"""
        super(RNNCellModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
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
        outputs = []
        emb = self.drop(self.encoder(input))
        for i, emb_t in enumerate(emb.chunk(emb.size(1), dim=1)):
            h_t, c_t = self.rnn(emb_t, (h_t, c_t))
            outputs += [h_t]
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTMCell':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class RNNModelNoEnc(nn.Module):
    """Container module with a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, nhid, nlayers, dropout=0.1, attention=False):
        super(RNNModelNoEnc, self).__init__()
        self.drop = nn.Dropout(dropout)
        # Choose attention model
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ntoken, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ntoken, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
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



class HRNNModel(nn.Module):
    """Hierarchical LSTM"""
    '''MUST take batch size of one, one layer (for now)'''
    # TODO different hyperparams (esp learning rate), type of model

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers=1, dropout=0.1):
        super(HRNNModel, self).__init__()
	nlayers = 1
        self.low = RNNModel(rnn_type, ntoken, ninp, nhid, nlayers, dropout)
        self.high = RNNModelNoEnc(rnn_type, nhid, nhid, nlayers, dropout) 

    def train(self, is_test=False):
	super(HRNNModel, self).train(is_test)
	self.low.train(is_test)
	self.high.train(is_test)

    def zero_grad(self):
	super(HRNNModel, self).zero_grad()
	self.low.zero_grad()
	self.high.zero_grad()

    # TODO Better to just loop.
    # TODO Which in the tuple from an LSTM should you get?
    # Dude this is a computation graph, everything needs to be an op.
    def forward_recurse(self, inp, hidden_low, hidden_high):
	# new measure
	if (inp.size()[0] == 1):
	    step_inp = inp.narrow(0, 0, 1) 
            step_out, _ = self.low(step_inp, hidden_low)
	    return step_out
        if (inp[0].data == 2).all():
            hidden_low_one, hidden_high = self.high(hidden_low[0].view(1,1,-1), hidden_high)
	    hidden_low = (hidden_low_one, hidden_low[1])
        
	step_inp = inp.narrow(0, 0, 1)
        step_out, hidden_low = self.low(step_inp, hidden_low)
        next_inp = inp.narrow(0, 1, inp.size()[0]-1)
        return torch.cat((step_out, self.forward_recurse(next_inp, hidden_low, hidden_high)))

    def forward(self, input, hiddens):
	hidden_low, hidden_high = hiddens
	'''
	out = []
	for i in range(input.size()[0]):
	    step_inp = inp.narrow(0, i, i+1) 
     	    step_out, hidden_low = self.low(step_inp, hidden_low)
	    out.append(step_out)
	return torch.LongTensor(out)
	'''

        return self.forward_recurse(input, hidden_low, hidden_high), (hidden_low, hidden_high)

    def init_hidden(self, bsz):
        return self.low.init_hidden(bsz), self.high.init_hidden(bsz)

