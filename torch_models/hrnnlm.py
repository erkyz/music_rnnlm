import torch
import torch.nn as nn
from torch.autograd import Variable
import time, util

import rnnlm

class HRNNModel(nn.Module):
    """Hierarchical LSTM"""
    # TODO different hyperparams (esp learning rate), type of model

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers=1, dropout=0.1):
        super(HRNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.low = getattr(nn, rnn_type+'Cell')(ninp+nhid, nhid, nlayers)
            self.high = getattr(nn, rnn_type+'Cell')(nhid, nhid, nlayers)
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

    def forward(self, input, hiddens, measure_token_idx):
        output = []
        hidden_low, hidden_high = hiddens
        h_low_t, c_low_t = hidden_low
        h_high_t, c_high_t = hidden_high
        for i, x_t in enumerate(input.chunk(input.size(1), dim=1)):
            # TODO make these faster.
            filter_t = Variable((x_t.data == measure_token_idx).expand(
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



class FactorHRNNModel(nn.Module):
    """Hierarchical LSTM"""
    def __init__(self, rnn_type, ntokens, ninp, nhid, nlayers=1, dropout=0.1):
        super(FactorHRNNModel, self).__init__()
        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], ninp))
        if rnn_type in ['LSTM', 'GRU']:
            self.low = getattr(nn, rnn_type+'Cell')(
                ninp*self.num_channels+nhid, nhid, nlayers)
            self.high = getattr(nn, rnn_type+'Cell')(nhid, nhid, nlayers)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(nhid, ntokens[i])) 
        self.encoders = rnnlm.AttrProxy(self, 'encoder_') 
        self.decoders = rnnlm.AttrProxy(self, 'decoder_') 

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

    def forward(self, input, hiddens, measure_token_idx):
        output = []
        hidden_low, hidden_high = hiddens
        h_low_t, c_low_t = hidden_low
        h_high_t, c_high_t = hidden_high
        chunked = [input[i].chunk(input[i].size(1), dim=1) for i in range(len(input))]
        batch_size = input[0].size(0)
        for i, x_t in enumerate(zip(*chunked)):
            first_channel = x_t[0]
            filter_t = Variable((first_channel.data == measure_token_idx).expand(
                batch_size, h_high_t.size(1)).type(torch.cuda.FloatTensor))
            embs = []
            for i in range(self.num_channels):
                embs.append(self.drop(self.encoders[i](torch.t(x_t[i])))) 
            embs_cat = torch.cat(embs, dim=2).view(batch_size,1,-1)
            inp_t = torch.cat([embs_cat, h_high_t.view(first_channel.size(0),1,-1)], 2)
            h_low_t, c_low_t = self.low(inp_t, (h_low_t, c_low_t))
            h_high_tmp, c_high_tmp = self.high(h_low_t, (h_high_t, c_high_t))
            h_high_t = torch.mul(filter_t, h_high_tmp) + torch.mul(1-filter_t, h_high_t)
            c_high_t = torch.mul(filter_t, c_high_tmp) + torch.mul(1-filter_t, c_high_t)
            output += [h_low_t]
        output = torch.stack(output, 1).squeeze(2)
        decs = []
        for i in range(self.num_channels):
            decoded = self.decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))

        return decs, ((h_low_t, c_low_t), (h_high_t, c_high_t))

    
    # returns tuple for hidden states of low and high LSTMs
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return ((Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_())),
                    (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_())))


