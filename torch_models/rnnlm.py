import torch
import torch.nn as nn
from torch.autograd import Variable
import time, util

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class RNNModel(nn.Module):
    def __init__(self, args):
        '''
        ntokens is a list of the numbers of tokens in the dictionary of each channel
        '''
        super(RNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens

        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        if args.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.rnn_type)(
                args.emsize*self.num_channels, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[args.rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(args.emsize, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        for i in range(self.num_channels):
            self.encoders[i].weight.data.uniform_(-initrange, initrange)
            self.decoders[i].bias.data.fill_(0)
            self.decoders[i].weight.data.uniform_(-initrange, initrange)

    def forward(self, data, hidden):
        ''' input should be a list with aligned inputs for each channel '''
        ''' returns a list of outputs for each channel '''
        inputs = data["data"]
        embs = []
        batch_size = inputs[0].size(0)
        for i in range(self.num_channels):
            embs.append(self.drop(self.encoders[i](torch.t(inputs[i])))) 
        rnn_input = torch.cat(embs, dim=2) 
        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lens, batch_first=True)
        output, hidden = self.rnn(rnn_input, hidden)
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.drop(torch.transpose(output, 0, 1))
        decs = []
        for i in range(self.num_channels):
            decoded = self.decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
        return decs, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())



class CRNNModel(nn.Module):
    ''' Conditional RNN: in this case, we concat the conditions to the RNN input '''
    # TODO OOP.
    def __init__(self, args):
        super(CRNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens

        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.num_conditions = args.num_conditions
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        # conditions encoder
        self.add_module('encoder_' + str(self.num_channels), nn.Embedding(args.num_conditions, args.emsize))
        if args.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.rnn_type)(
                args.emsize*(self.num_channels+1), nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[args.rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(args.emsize, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        for i in range(self.num_channels):
            self.encoders[i].weight.data.uniform_(-initrange, initrange)
            self.decoders[i].bias.data.fill_(0)
            self.decoders[i].weight.data.uniform_(-initrange, initrange)

    def forward(self, data, hidden):
        ''' input should be a list with aligned inputs for each channel '''
        ''' conditions should be a list with aligned inputs for each channel '''
        ''' returns a list of outputs for each channel '''

        inputs = data["data"]
        conditions = data["conditions"]
        emb_conds = []
        batch_size = inputs[0].size(0)
        for i in range(self.num_channels):
            # conditions[i] is batch_size x seqlen
            emb_c = self.encoders[self.num_channels](torch.t(conditions[i])) 
            # inputs[i] is batch_size x seqlen
            # emb should be seqlen x batch_size x args.emsize
            emb = self.encoders[i](torch.t(inputs[i]))
            # concat note embedding with condition embedding for this channel
            emb_cond = torch.cat([emb, emb_c], dim=2)
            emb_conds.append(self.drop(emb_cond)) 
        rnn_input = torch.cat(emb_conds, dim=2) 
        # rnn_input = torch.cat([rnn_input, torch.t(conditions[i]).contiguous().view(conditions[i].size(0),batch_size,1)], dim=2)
        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lens, batch_first=True)
        hidden = self.rnn(rnn_input, hidden)
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.drop(torch.transpose(output, 0, 1))
        decs = []
        for i in range(self.num_channels):
            decoded = self.decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
        return decs, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())



