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


class RNNCellModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, args):
        super(RNNCellModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        ntokens = args.ntokens
        dropout = args.dropout

        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.num_conditions = args.num_conditions
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        if args.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.rnn_type + 'Cell')(
                args.emsize*self.num_channels, nhid, nlayers)
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
        inputs = data["data"]
        output = []
        h_t, c_t = hidden
        embs = []
        batch_size = inputs[0].size(0)
        for i in range(self.num_channels):
            embs.append(self.drop(self.encoders[i](inputs[i])))
        rnn_input = torch.cat(embs, dim=2)
        for i, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
            h_t, c_t = self.rnn(emb_t.squeeze(1), (h_t, c_t))
            # TODO this only works for LSTMCell
            output += [h_t]
        output = torch.stack(output, 1).squeeze(0)
        output = self.drop(output)

        decs = []
        for i in range(self.num_channels):
            decoded = self.decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
        return decs, (h_t, c_t)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())


# TODO rename
class XRNNModel(nn.Module):
    ''' Conditional RNN: in this case, we concat the conditions to the RNN input '''
    # TODO OOP.
    def __init__(self, args):
        super(XRNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens

        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.hsize = nhid
        self.total_nhid = 2*nhid # because we concat 2 together
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        if args.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.rnn_type + 'Cell')(
                args.emsize*self.num_channels, self.total_nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[args.rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(args.emsize, self.total_nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(self.total_nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        for i in range(self.num_channels):
            self.encoders[i].weight.data.uniform_(-initrange, initrange)
            self.decoders[i].bias.data.fill_(0)
            self.decoders[i].weight.data.uniform_(-initrange, initrange)

    def forward(self, data, hidden):
        ''' input should be a list with aligned inputs for each channel '''
        ''' conditions should be a list with indices of the prev hidden states '''
        ''' for conditioning. -1 means no condition '''
        ''' returns a list of outputs for each channel '''
        # list of (bsz,seqlen)
        inputs = data["data"]
        # data["conditions"] is a list of (bsz,seqlen)
        conditions = data["conditions"][0].data.tolist() # TODO
        output = []
        batch_size = inputs[0].size(0)
        embs = []
        for c in range(self.num_channels):
            embs.append(self.drop(self.encoders[c](inputs[c])))
        rnn_input = torch.cat(embs, dim=2)
        prev_hs = [hidden]
        to_concat = (Variable(torch.cuda.FloatTensor(batch_size, self.hsize).zero_()),
                    Variable(torch.cuda.FloatTensor(batch_size, self.hsize).zero_()))
        for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
            # LSTMCell TODO GRU
            tmp = []
            for state_idx in range(2):
                for b in range(batch_size):
                    to_concat[state_idx][b].data.copy_(prev_hs[conditions[b][t]][state_idx][b].data)
                # The trick here is that if we don't want to concat with a previous
                # state, conditions[i] == -1.
                tmp.append(torch.cat(
                    [prev_hs[-1][state_idx], to_concat[state_idx]],
                    dim=1))
            hidden = self.rnn(emb_t.squeeze(1), (tmp[0], tmp[1]))
            # For now, we're going to save all the prev_hs. if this is slow, we won't.
            prev_hs.append((hidden[0][:,:self.hsize], hidden[1][:,:self.hsize]))
            output += [hidden[0]]
        output = torch.stack(output, 1).squeeze(0)
        output = self.drop(output)

        decs = []
        for i in range(self.num_channels):
            decoded = self.decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
        return decs, prev_hs[-1]

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        # This is a little strange because the model actually has nhid units
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.hsize).zero_()),
                    Variable(weight.new(bsz, self.hsize).zero_()))
        else:
            return Variable(weight.new(bsz, self.hsize).zero_())



