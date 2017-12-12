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
    def __init__(self, rnn_type, ntokens, emsize, nhid, nlayers, dropout=0.1):
        '''
        ntokens is a list of the numbers of tokens in the dictionary of each channel
        '''
        super(RNNModel, self).__init__()
        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], emsize)) 
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                emsize*self.num_channels, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emsize, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        for i in range(self.num_channels):
            self.encoders[i].weight.data.uniform_(-initrange, initrange)
            self.decoders[i].bias.data.fill_(0)
            self.decoders[i].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        ''' input should be a list with aligned inputs for each channel '''
        ''' returns a list of outputs for each channel '''
        embs = []
        batch_size = inputs[0].size(0)
        for i in range(self.num_channels):
            embs.append(self.drop(self.encoders[i](torch.t(inputs[i])))) 
        rnn_input = torch.cat(embs, dim=2) 
        '''
        for i in range(self.num_conditions):
            rnn_input = torch.cat([rnn_input, torch.t(conditions[i]).contiguous().view(conditions[i].size(0),batch_size,1)], dim=2)
        '''
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


class RNNCellModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntokens, emsize, nhid, nlayers, dropout=0.1):
        super(RNNCellModel, self).__init__()
        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], emsize)) 
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type + 'Cell')(
                emsize*self.num_channels, nhid, nlayers)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        for i in range(self.num_channels):
            self.encoders[i].weight.data.uniform_(-initrange, initrange)
            self.decoders[i].bias.data.fill_(0)
            self.decoders[i].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
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

class RRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntokens, emsize, nhid, nlayers, dropout=0.1):
        super(RRNNModel, self).__init__()
        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], emsize)) 
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type + 'Cell')(
                emsize*self.num_channels, nhid, nlayers)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = rnn_type
        self.emsize = emsize
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntokens = ntokens
        self.temperature = 0.9 # TODO don't hardcode here

    def init_weights(self):
        initrange = 0.1
        for i in range(self.num_channels):
            self.encoders[i].weight.data.uniform_(-initrange, initrange)
            self.decoders[i].bias.data.fill_(0)
            self.decoders[i].weight.data.uniform_(-initrange, initrange)
        self.gamma = torch.autograd.Variable(torch.Tensor([0]).cuda().float())

    def forward(self, inputs, hidden, measure_weights, measure_length, measure_steps):
        '''
        |inputs|: each measure is padded to the same length
        |measure_weights| is bs x (seqlen / measure_length) x (seqlen / measure_length), 
            the "attention" weights
        |measure_length| is an integer specifying how many tokens are in each measure.
        |measure_steps| is a list, for each channel, of a list with length measure_length 
            of tensors bs x num_measures, the ith event in each measure
        '''
        output = []
        lookback_output = []
        h_t, c_t = hidden
        batch_size = inputs[0].size(0)
        num_measures = inputs[0].size(1) / measure_length
         
        # lookback embedding: the weighted sum of the embeddings at this step
        # in the previous measures.
        lookback_embeddings = [] # Should be [num_channels, num_measures * measure_length]
        for i in range(self.num_channels):
            lookback_embeddings.append([-1]*measure_length)
        for curr_measure in range(num_measures):
            if curr_measure == 0:
                continue
            # TODO vary the temperature as curr_measure increases.
            # normalized previous measure weights
            softmax = torch.nn.functional.softmax(
                measure_weights[:,curr_measure,:curr_measure].permute(1,0) \
                / self.temperature).permute(1,0)
            tiled_softmax = softmax.unsqueeze(2).expand(
                    batch_size, curr_measure, self.emsize * self.num_channels)
            for step in range(measure_length):
                lookback_channel_embedding = []
                for channel in range(self.num_channels):
                    # bs x curr_measure
                    prev_steps = measure_steps[channel][step][:,:curr_measure]
                    # bs x curr_measure x emsize
                    prev_embs = self.encoders[channel](prev_steps)
                    # weighted sum: bs x emsize
                    lookback_embeddings[channel].append(
                        torch.sum(torch.mul(tiled_softmax, prev_embs), dim=1))
        embs = []
        for i in range(self.num_channels):
            embs.append(self.drop(self.encoders[i](inputs[i])))
        rnn_input = torch.cat(embs, dim=2)
        for i, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
            curr_measure = int(i / num_measures)
            # TODO this only works for LSTMCell
            h_t, c_t = self.rnn(emb_t.squeeze(1), (h_t, c_t))
            output += [h_t]
            step_lookback_output = []
            for channel in range(self.num_channels):
                if curr_measure == 0:
                    step_lookback_output.append(torch.autograd.Variable(torch.zeros(batch_size, self.ntokens[channel], 1).type(torch.cuda.FloatTensor), requires_grad=False))
                else:
                    # dot product for similarities
                    step_lookback_output.append(
                        torch.bmm(self.encoders[channel].weight.expand(batch_size, self.ntokens[channel], self.emsize),
                                     lookback_embeddings[channel][i].unsqueeze(2)))
            # concat over the channels
            lookback_output += [torch.cat(step_lookback_output, dim=1).squeeze(2)]
        output = torch.stack(output, 1).squeeze(0)
        output = self.drop(output)
        lookback_output = torch.stack(lookback_output, 1).squeeze(0)

        decs = []
        for i in range(self.num_channels):
            decoded = self.decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append((1-self.gamma) * decoded.view(output.size(0), output.size(1), decoded.size(1)) \
                         + self.gamma * lookback_output)
        return decs, (h_t, c_t)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())


