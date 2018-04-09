import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
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
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.encoders[i].weight.data)
            nn.init.xavier_normal(self.decoders[i].weight.data)
            self.decoders[i].bias.data.fill_(0)

    # scheduled sampler
    def forward_ss(self, data, hidden, args):
        inputs = data["data"]
        batch_size = inputs[0].size(0)
        # linear annealing 
        prob_gold = max(float(args.epochs-args.epoch)/args.epochs, 0)
        decs = [[]*self.num_channels]

        tmp = []
        for c in range(self.num_channels):
            tmp.append(self.drop(self.encoders[c](inputs[c][:,0])))
        emb_t = torch.cat(tmp, dim=1)

        for t in range(inputs[0].size(1)):
            hidden = self.rnn(emb_t, hidden)
            out_t = hidden[0] if self.rnn_type == 'LSTM' else hidden
            tmp = []
            for c in range(self.num_channels):
                decoded = self.decoders[c](out_t)
                decs[c].append(decoded.unsqueeze(1))
                if random.random() > prob_gold and t >= args.skip_first_n_note_losses:
                    d = decoded.data.div(args.temperature)
                    weights = torch.stack([F.softmax(d[b]) for b in range(batch_size)], 0)
                    sampled_idxs = torch.multinomial(weights, 1)
                    idxs = Variable(sampled_idxs.squeeze().data)
                else:
                    idxs = inputs[c][:,t]
                tmp.append(self.drop(self.encoders[c](idxs)))
            emb_t = torch.cat(tmp, dim=1)

        for c in range(self.num_channels):
            decs[c] = torch.cat(decs[c], dim=1)

        return decs, hidden

    def forward(self, data, hidden, args):
        if args.ss:
            return self.forward_ss(data, hidden, args)
        else:
            inputs = data["data"]
            output = []
            embs = []
            batch_size = inputs[0].size(0)
            for i in range(self.num_channels):
                embs.append(self.drop(self.encoders[i](inputs[i])))
            rnn_input = torch.cat(embs, dim=2)
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                # emb_t is [bsz x 1 x emsize]
                hidden = self.rnn(emb_t.squeeze(1), hidden)
                output += [hidden[0]] if self.rnn_type == 'LSTM' else [hidden]
            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
            return decs, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())


class XRNNModel(nn.Module):
    # TODO OOP.
    def __init__(self, args):
        super(XRNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens

        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.nhid = nhid # output dimension
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        if args.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.rnn_type + 'Cell')(
                args.emsize*self.num_channels, self.nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[args.rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(args.emsize, self.nhid, nlayers, nonlinearity=nonlinearity)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nlayers = nlayers
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.A = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid).zero_())
        nn.init.xavier_normal(self.A)
        self.B = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid).zero_())
        nn.init.xavier_normal(self.B)
        self.default_h = nn.Parameter(torch.FloatTensor(self.nhid).zero_())
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.encoders[i].weight.data)
            nn.init.xavier_normal(self.decoders[i].weight.data)
            self.decoders[i].bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, 2*self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())

    def get_new_h_t(self, prev_hs, conditions, batch_size, t, args):
        to_concat = []
        for b in range(batch_size):
            prev_idx = conditions[b][t]
            # prev = prev_hs[prev_idx+1][b] if t > args.skip_first_n_note_losses and prev_idx != -1 and prev_idx != t-1 else self.default_h
            prev = prev_hs[prev_idx][b] if t > args.skip_first_n_note_losses and prev_idx != -1 else self.default_h
            # prev = self.default_h
            l = torch.matmul(self.A, prev_hs[-1][b])
            r = torch.matmul(self.B, prev)
            to_concat.append(l.unsqueeze(0)+r.unsqueeze(0))
        return torch.cat(to_concat, dim=0)


    def forward_ss(self, data, hidden, args, prev_hs=None):
        # linear annealing 
        prob_gold = max(float(args.epochs-args.epoch)/args.epochs, 0)

        # list of (bsz,seqlen)
        inputs = data["data"]
        # data["conditions"] is a list of (bsz,seqlen)
        conditions = data["conditions"][0].data.tolist() # TODO
        batch_size = inputs[0].size(0)
        if prev_hs is None:
            # Train mode
            prev_hs = [hidden[0] if self.rnn_type == 'LSTM' else hidden]
                 
        decs = [[]*self.num_channels]
        tmp = []
        for c in range(self.num_channels):
            tmp.append(self.drop(self.encoders[c](inputs[c][:,0])))
        emb_t = torch.cat(tmp, dim=1)
        for t in range(inputs[0].size(1)):
            new_h_t = self.get_new_h_t(prev_hs, conditions, batch_size, t, args)
            if self.rnn_type == 'LSTM': 
                hidden = self.rnn(emb_t.squeeze(1), (new_h_t, hidden[1]))
            else:
                hidden = self.rnn(emb_t.squeeze(1), new_h_t)
            out_t = hidden[0] if self.rnn_type == 'LSTM' else hidden
            prev_hs.append(out_t)
            out_t = self.drop(out_t)
            tmp = []
            for c in range(self.num_channels):
                decoded = self.decoders[c](out_t)
                decs[c].append(decoded.unsqueeze(1))
                weights = torch.stack([F.softmax(decoded.data.div(args.temperature)[i]) for i in range(batch_size)], 0) 
                sampled_idxs = torch.multinomial(weights, 1)
                idx = inputs[c][:,t] if random.random() < prob_gold else sampled_idxs.squeeze()
                tmp.append(self.drop(self.encoders[c](idx)))
            emb_t = torch.cat(tmp, dim=1)

        for c in range(self.num_channels):
            decs[c] = torch.cat(decs[c], dim=1)

        return decs, hidden


    def forward(self, data, hidden, args, prev_hs=None):
        ''' input should be a list with aligned inputs for each channel '''
        ''' conditions should be a list with indices of the prev hidden states '''
        ''' for conditioning. -1 means no condition '''
        ''' returns a list of outputs for each channel '''
        # print self.default_h
        if args.ss:
            return self.forward_ss(data, hidden, args, prev_hs)
        else:
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
            if prev_hs is None:
                # Train mode
                prev_hs = [hidden[0] if self.rnn_type == 'LSTM' else hidden]
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                new_h_t = self.get_new_h_t(prev_hs, conditions, batch_size, t, args)
                if self.rnn_type == 'LSTM': 
                    hidden = self.rnn(emb_t.squeeze(1), (new_h_t, hidden[1]))
                else:
                    hidden = self.rnn(emb_t.squeeze(1), new_h_t)
                prev_hs.append(hidden[0] if self.rnn_type == 'LSTM' else hidden)
                output += [prev_hs[-1]]
            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
            return decs, prev_hs[-1]





class ERNNModel(nn.Module):
    def __init__(self, args):
        super(ERNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens

        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.nhid = nhid # output dimension
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        self.rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        self.parallel_rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nlayers = nlayers
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.A = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid).zero_())
        nn.init.xavier_normal(self.A)
        self.B = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid).zero_())
        nn.init.xavier_normal(self.B)
        self.default_h = nn.Parameter(torch.FloatTensor(self.nhid).zero_())
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.encoders[i].weight.data)
            nn.init.xavier_normal(self.decoders[i].weight.data)
            self.decoders[i].bias.data.fill_(0)

    def is_lstm(self):
        return self.rnn_type == 'LSTM'

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.is_lstm():
            main_hidden= (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, 2*self.nhid).zero_()))
            parallel_hidden= (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, 2*self.nhid).zero_()))
        else:
            main_hidden = Variable(weight.new(bsz, self.nhid).zero_())
            parallel_hidden = Variable(weight.new(bsz, self.nhid).zero_())
        return {'main': main_hidden, 'parallel': parallel_hidden}
    
    def get_new_h_t(self, curr_h, prev_hs, conditions, batch_size, t, args):
        to_concat = []
        for b in range(batch_size):
            prev_idx = conditions[b][t]
            '''
            w = 1
            use_prev = t > args.skip_first_n_note_losses and prev_idx != -1 and prev_idx < t-w
            prev = prev_hs[prev_idx+w][b] if use_prev else self.default_h
            '''
            prev = self.default_h
            l = torch.matmul(self.A, curr_h[b])
            r = torch.matmul(self.B, prev)
            to_concat.append(l.unsqueeze(0)+r.unsqueeze(0))
        return torch.cat(to_concat, dim=0)


    # TODO implement this for PRNN
    def forward_ss(self, data, hidden, args, prev_hs=None):
         return 

    def forward(self, data, hidden, args, prev_hs=None):
        # hidden is a dict
        if args.ss:
            return self.forward_ss(data, hidden, args, prev_hs)
        else:
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
            if prev_hs is None:
                # Train mode
                prev_hs = [hidden["parallel"][0] if self.is_lstm() else hidden["parallel"]]
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                main_h = hidden["main"]
                parallel_h = hidden["parallel"]
                curr_main_h = main_h[0] if self.is_lstm() else main_h
                new_h_t = self.get_new_h_t(curr_main_h, prev_hs, conditions, batch_size, t, args)
                if self.is_lstm():
                    hidden["main"] = self.rnn(emb_t.squeeze(1), (new_h_t, main_h[1]))
                    hidden["parallel"] = self.parallel_rnn(emb_t.squeeze(1), parallel_h)
                else:
                    hidden["main"] = self.rnn(emb_t.squeeze(1), new_h_t)
                    hidden["parallel"] = self.rnn(emb_t.squeeze(1), parallel_h)
                prev_hs.append(parallel_h[0] if self.is_lstm() else parallel_h)
                output += [hidden["main"][0] if self.is_lstm() else hidden["main"]]
            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
            return decs, hidden






class PRNNModel(nn.Module):
    def __init__(self, args):
        super(PRNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens

        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.nhid = nhid # output dimension
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        self.rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        self.parallel_rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nlayers = nlayers
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.A = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid).zero_())
        nn.init.xavier_normal(self.A)
        self.B = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid).zero_())
        nn.init.xavier_normal(self.B)
        self.default_h = nn.Parameter(torch.FloatTensor(self.nhid).zero_())
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.encoders[i].weight.data)
            nn.init.xavier_normal(self.decoders[i].weight.data)
            self.decoders[i].bias.data.fill_(0)

    def is_lstm(self):
        return self.rnn_type == 'LSTM'

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.is_lstm():
            main_hidden= (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, 2*self.nhid).zero_()))
            parallel_hidden= (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, 2*self.nhid).zero_()))
        else:
            main_hidden = Variable(weight.new(bsz, self.nhid).zero_())
            parallel_hidden = Variable(weight.new(bsz, self.nhid).zero_())
        return {'main': main_hidden, 'parallel': parallel_hidden}
    
    def get_new_h_t(self, curr_h, prev_hs, conditions, batch_size, t, args):
        to_concat = []
        for b in range(batch_size):
            prev_idx = conditions[b][t]
            '''
            w = 1
            use_prev = t > args.skip_first_n_note_losses and prev_idx != -1 and prev_idx < t-w
            prev = prev_hs[prev_idx+w][b] if use_prev else self.default_h
            '''
            prev = self.default_h
            l = torch.matmul(self.A, curr_h[b])
            r = torch.matmul(self.B, prev)
            to_concat.append(l.unsqueeze(0)+r.unsqueeze(0))
        return torch.cat(to_concat, dim=0)


    # TODO implement this for PRNN
    def forward_ss(self, data, hidden, args, prev_hs=None):
         return 

    def forward(self, data, hidden, args, prev_hs=None):
        # hidden is a dict
        if args.ss:
            return self.forward_ss(data, hidden, args, prev_hs)
        else:
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
            if prev_hs is None:
                # Train mode
                prev_hs = [hidden["parallel"][0] if self.is_lstm() else hidden["parallel"]]
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                main_h = hidden["main"]
                parallel_h = hidden["parallel"]
                curr_main_h = main_h[0] if self.is_lstm() else main_h
                new_h_t = self.get_new_h_t(curr_main_h, prev_hs, conditions, batch_size, t, args)
                if self.is_lstm():
                    hidden["main"] = self.rnn(emb_t.squeeze(1), (new_h_t, main_h[1]))
                    hidden["parallel"] = self.parallel_rnn(emb_t.squeeze(1), parallel_h)
                else:
                    hidden["main"] = self.rnn(emb_t.squeeze(1), new_h_t)
                    hidden["parallel"] = self.rnn(emb_t.squeeze(1), parallel_h)
                prev_hs.append(parallel_h[0] if self.is_lstm() else parallel_h)
                output += [hidden["main"][0] if self.is_lstm() else hidden["main"]]
            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
            return decs, hidden




class VineRNNModel(nn.Module):
    ''' Conditional RNN: in this case, we concat the conditions to the RNN input '''
    def __init__(self, args):
        super(VineRNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens

        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.nhid = nhid
        for i in range(self.num_channels):
            self.add_module('encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        if args.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.rnn_type + 'Cell')(
                args.emsize*self.num_channels, self.nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[args.rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(args.emsize, self.nhid, nlayers, nonlinearity=nonlinearity)
        for i in range(len(ntokens)):
            self.add_module('decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        self.encoders = AttrProxy(self, 'encoder_') 
        self.decoders = AttrProxy(self, 'decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nlayers = nlayers

    def init_weights(self):
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.encoders[i].weight.data)
            nn.init.xavier_normal(self.decoders[i].weight.data)
            self.decoders[i].bias.data.fill_(0)

    def forward(self, data, hidden, prev_hs=None):
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
        if prev_hs is None:
            prev_hs = [hidden]
        for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
            to_cat = []
            for b in range(batch_size):
                # TODO make this work for LSTMs
                to_cat.append(torch.unsqueeze(prev_hs[conditions[b][t]][b], 0))
            new_h = torch.cat(to_cat, dim=0)
            hidden = self.rnn(emb_t.squeeze(1), new_h)
            prev_hs.append(hidden)
            output += [hidden[0] if self.rnn_type == 'LSTM' else hidden]
        output = torch.stack(output, 1)
        output = self.drop(output)

        decs = []
        for i in range(self.num_channels):
            decoded = self.decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
        return decs, prev_hs[-1]

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())


