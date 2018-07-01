import torch
import torch.nn as nn
from torch.autograd import Variable
import time

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class RNNModel(nn.Module):
    def __init__(self, args):
        super(RNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        # |ntokens| is a list of the numbers of tokens in the dictionary of each channel
        ntokens = args.ntokens

        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.rnn_type = args.rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        # Embedding "encoder"
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
        # Embedding "decoder"
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(nhid, ntokens[i])) 
        
        # Access like a list, where index is the channel idx.
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

    @property
    def need_conditions(self):
        return False

    def init_weights(self, *arg):
        # Xavier initialization
        for c in range(self.num_channels):
            nn.init.xavier_normal(self.encoders[c].weight.data)
            nn.init.xavier_normal(self.decoders[c].weight.data)
            self.decoders[c].bias.data.fill_(0)

    def forward(self, data, hidden, *arg):
        ''' input should be a list with aligned inputs for each channel '''
        ''' returns a list of outputs for each channel '''
        inputs = data["data"]
        embs = []
        batch_size = inputs[0].size(0)
        for i in range(self.num_channels):
            embs.append(self.drop(self.encoders[i](torch.t(inputs[i])))) 
        rnn_input = torch.cat(embs, dim=2) 
        # TODO use rnn.pack_padded_sequence to save computation?
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.drop(torch.transpose(output, 0, 1))
        decs = []
        for i in range(self.num_channels):
            decoded = self.decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
        return decs, hidden

    def init_hidden(self, bsz):
        # Zero initialization
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

# END RNNModel


class CRNNModel(RNNModel):
    '''
    Very simple conditional RNN where we embed the conditions and concat to the RNN input 
    '''
    def __init__(self, args):
        super(CRNNModel, self).__init__()

        self.num_conditions = args.num_conditions
        # Conditions encoder is added on as the last encoder in self.encoders.
        # We embed from the number of conditions into |args.emsize|
        self.add_module('encoder_' + str(self.num_channels), nn.Embedding(args.num_conditions, args.emsize))

    @property
    def need_conditions(self):
        return True

    def forward(self, data, hidden):
        '''
        data["data"] should be a list with aligned inputs for each channel 
        data["conditions"] should be a list with aligned inputs for each channel 
        
        Returns a list of outputs for each channel 
        '''

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
        hidden = self.rnn(rnn_input, hidden)
        output = self.drop(torch.transpose(output, 0, 1))
        decs = []
        for i in range(self.num_channels):
            decoded = self.decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
        return decs, hidden

# END CRNNModel
