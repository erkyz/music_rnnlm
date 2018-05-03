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
            self.add_module('emb_encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        if args.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.rnn_type + 'Cell')(
                args.emsize*self.num_channels, nhid, nlayers)
        for i in range(len(ntokens)):
            self.add_module('emb_decoder_' + str(i), nn.Linear(nhid, ntokens[i])) 
        self.emb_encoders = AttrProxy(self, 'emb_encoder_') 
        self.emb_decoders = AttrProxy(self, 'emb_decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.emb_encoders[i].weight.data)
            nn.init.xavier_normal(self.emb_decoders[i].weight.data)
            self.emb_decoders[i].bias.data.fill_(0)

    # scheduled sampler
    def forward_ss(self, data, hidden, args):
        inputs = data["data"]
        batch_size = inputs[0].size(0)
        # linear annealing 
        prob_gold = max(float(args.epochs-args.epoch)/args.epochs, 0)
        decs = [[]*self.num_channels]

        tmp = []
        for c in range(self.num_channels):
            tmp.append(self.drop(self.emb_encoders[c](inputs[c][:,0])))
        emb_t = torch.cat(tmp, dim=1)

        for t in range(inputs[0].size(1)):
            hidden = self.rnn(emb_t, hidden)
            out_t = hidden[0] if self.rnn_type == 'LSTM' else hidden
            tmp = []
            for c in range(self.num_channels):
                decoded = self.emb_decoders[c](out_t)
                decs[c].append(decoded.unsqueeze(1))
                if random.random() > prob_gold and t >= args.skip_first_n_note_losses:
                    d = decoded.data.div(args.temperature)
                    weights = torch.stack([F.softmax(d[b]) for b in range(batch_size)], 0)
                    sampled_idxs = torch.multinomial(weights, 1)
                    idxs = Variable(sampled_idxs.squeeze().data)
                else:
                    idxs = inputs[c][:,t]
                tmp.append(self.drop(self.emb_encoders[c](idxs)))
            emb_t = torch.cat(tmp, dim=1)

        for c in range(self.num_channels):
            decs[c] = torch.cat(decs[c], dim=1)

        return decs, hidden

    def forward(self, data, hidden, args, prev_data=None, curr_t=None):
        if args.ss:
            return self.forward_ss(data, hidden, args)
        else:
            inputs = data["data"]
            output = []
            embs = []
            batch_size = inputs[0].size(0)
            for i in range(self.num_channels):
                embs.append(self.drop(self.emb_encoders[i](inputs[i])))
            rnn_input = torch.cat(embs, dim=2)
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                # emb_t is [bsz x 1 x emsize]
                hidden = self.rnn(emb_t.squeeze(1), hidden)
                output += [hidden[0]] if self.rnn_type == 'LSTM' else [hidden]
            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.emb_decoders[i](
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


class AttentionRNNModel(nn.Module):
    def __init__(self, args):
        super(AttentionRNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens
         
        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.nhid = nhid # output dimension
        for i in range(self.num_channels):
            self.add_module('emb_encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        self.rnn = getattr(nn, args.rnn_type + 'Cell')(
            self.nhid+args.emsize*self.num_channels, self.nhid, nlayers) 
        self.fc1 = nn.Linear(self.nhid+args.emsize, self.nhid) 
        self.fc2 = nn.Linear(self.nhid, 1)

        for i in range(len(ntokens)):
            self.add_module('emb_decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        # For backbone RNN
        self.emb_encoders = AttrProxy(self, 'emb_encoder_') 
        self.emb_decoders = AttrProxy(self, 'emb_decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nlayers = nlayers

    def init_weights(self):
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.emb_encoders[i].weight.data)
            nn.init.xavier_normal(self.emb_decoders[i].weight.data)
            self.emb_decoders[i].bias.data.fill_(0)

    def is_lstm(self):
        return self.rnn_type == 'LSTM'

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        backbone = Variable(weight.new(bsz, self.nhid).zero_())
        return {'backbone': backbone}
   
    def get_self_attention_new_input(self, h_backbone, prev_data, args):
        # self-attention over previous segment encodings
        vs = []
        for i, prev_enc in enumerate(prev_data):
            x = torch.cat([prev_enc.squeeze(), h_backbone], dim=1)
            x = self.fc1(x)
            x = F.tanh(self.fc2(x))
            vs.append(x)
        softmax = F.softmax(torch.cat(vs, dim=1)) # bsz x t
        bsz = softmax.size(0)
        # This is not fast, but prev_data is a list for generate purposes (for now) 
        att_prev_enc = sum([torch.cat(
            [softmax[b][t]*prev_data[t][b] for b in range(bsz)]) for t in range(len(prev_data))])
        new_input = torch.cat([h_backbone, att_prev_enc.squeeze()], dim=1)
        return new_input

    # TODO implement this 
    def forward_ss(self, data, hidden, args, prev_data=None):
         return 

    def forward(self, data, hidden, args, prev_data=None, curr_t=None):
        # hidden is a dict
        # what we call "prev_data" is not actually previous hidden states.
        train_mode = curr_t is None
        if args.ss:
            return self.forward_ss(data, hidden, args, prev_data)
        else:
            # list of (bsz,seqlen)
            inputs = data["data"]
            bsz = inputs[0].size(0)
           
            output = []
            embs = []
            for c in range(self.num_channels):
                embs.append(self.drop(self.emb_encoders[c](inputs[c])))
            emb_start = self.emb_encoders[0](inputs[0][0][0])
            rnn_input = torch.cat(embs, dim=2)

            if train_mode:
                # Train mode. Need to save this as a list because of generate-mode.
                prev_data = []
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                # Note that t is input-indexed
                if curr_t is not None: t = curr_t
                # We initialize the score_softmax to be such that the model prefers 
                # the backbone RNN in the first measure. TODO magic number
                # Encoding is based on input indexing, not outputs.
                rnn_input_to_concat = []
                
                # In generate mode, t=0 even though we're at t=t, so index |inputs| with 0
                t_to_use = t if train_mode else 0

                prev_data.append(emb_t)
                rnn_input_t = self.get_self_attention_new_input(hidden['backbone'], prev_data, args)
                hidden['backbone'] = self.rnn(rnn_input_t, hidden['backbone'])
                output += [hidden['backbone']]

            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.emb_decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
            return decs, hidden

# end AttentionRNNModel 


class MRNNModel(nn.Module):
    def __init__(self, args):
        super(MRNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens
         
        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.nhid = nhid # output dimension
        for i in range(self.num_channels):
            self.add_module('emb_encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
            self.add_module('prev_emb_encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
            self.add_module('prev_emb_encoder_2_' + str(i), nn.Embedding(ntokens[i], args.emsize))
        self.rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        self.prev_enc_rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        self.prev_dec_rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        self.fc1 = nn.Linear(self.nhid+1, self.nhid) # +1 for the simscore
        self.fc2 = nn.Linear(self.nhid, 1)
        # self.fc3 = nn.Linear(self.nhid*2+1, self.nhid) # +1 for score_softmax
        self.fc3 = nn.Linear(1, self.nhid/2)
        self.fc4 = nn.Linear(self.nhid/2, 1)

        for i in range(len(ntokens)):
            self.add_module('emb_decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        # For backbone RNN
        self.emb_encoders = AttrProxy(self, 'emb_encoder_') 
        self.emb_decoders = AttrProxy(self, 'emb_decoder_') 
        # For prev_enc RNN
        self.prev_emb_encoders = AttrProxy(self, 'prev_emb_encoder_') 
        # For prev_dec RNN
        self.prev_emb_encoders_2 = AttrProxy(self, 'prev_emb_encoder_2_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nlayers = nlayers

    def init_weights(self):
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.emb_encoders[i].weight.data)
            nn.init.xavier_normal(self.prev_emb_encoders[i].weight.data)
            nn.init.xavier_normal(self.prev_emb_encoders_2[i].weight.data)
            nn.init.xavier_normal(self.emb_decoders[i].weight.data)
            self.emb_decoders[i].bias.data.fill_(0)

    def is_lstm(self):
        return self.rnn_type == 'LSTM'

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        backbone = Variable(weight.new(bsz, self.nhid).zero_())
        prev_enc = [Variable(weight.new(1, self.nhid).zero_()) for b in range(bsz)]
        prev_dec = [Variable(weight.new(1, self.nhid).zero_()) for b in range(bsz)]
        return {'backbone': backbone, 'prev_enc': prev_enc, 'prev_dec': prev_dec}
   
    def get_new_output(self, h_backbone, h_dec, score_softmax, args):
        # x = torch.cat([h_backbone.squeeze(), h_dec.squeeze(), score_softmax])
        x = score_softmax
        x = self.fc3(x) 
        x = self.fc4(x)
        alpha = F.sigmoid(x)
        if random.random() < 0.01:
            print score_softmax.data[0], alpha.data[0]
        return alpha*h_dec + (1-alpha)*h_backbone

    def self_attention(self, h_backbone, prev_encs, prev_weighted_score, args):
        # self-attention over previous segment encodings
        vs = []
        for i, prev_enc in enumerate(prev_encs):
            # Get the similarity score of the ith previous segment
            if args.cuda:
                s = Variable(torch.cuda.FloatTensor([prev_weighted_score[i]]), requires_grad=False)
            else:
                s = Variable(torch.FloatTensor([prev_weighted_score[i]]), requires_grad=False)
            # x = torch.cat([h_backbone, prev_enc.squeeze(), s])
            x = torch.cat([prev_enc.squeeze(), s])
            x = self.fc1(x) 
            x = F.tanh(self.fc2(x))
            vs.append(x)
        softmax = F.softmax(torch.cat(vs))
        decoder_h0 = torch.sum(
                torch.cat(
                    [prev_encs[i]*softmax[i] for i in range(len(prev_encs))]),
                0).unsqueeze(0)
        score_softmax = torch.sum(torch.cat(
            [torch.mul(
                Variable(torch.cuda.FloatTensor([prev_weighted_score[i]]) if args.cuda 
                    else torch.FloatTensor([prev_weighted_score[i]]), requires_grad=False),
                softmax[i]
                ) for i in range(len(prev_encs))]
            ), 0)
        return decoder_h0, score_softmax

    # TODO implement this 
    def forward_ss(self, data, hidden, args, prev_data=None):
         return 

    def forward(self, data, hidden, args, prev_data=None, curr_t=None):
        # hidden is a dict
        # what we call "prev_data" is not actually previous hidden states.
        train_mode = curr_t is None
        if args.ss:
            return self.forward_ss(data, hidden, args, prev_data)
        else:
            # list of (bsz,seqlen)
            inputs = data["data"]
            bsz = inputs[0].size(0)
            conditions = data["conditions"][0]
            segs = data["metadata"][0]
            # NOTE segs is indexed ignoring the START token, so it's aligned with OUTPUTS!
            beg_idxs = [[s[0] for s in segs[b]] for b in range(bsz)]
           
            # print inputs[0][0], conditions[0]
            # print "-"*88
            weight = next(self.parameters()).data
            output = []
            embs = []
            for c in range(self.num_channels):
                embs.append(self.drop(self.emb_encoders[c](inputs[c])))
            emb_start = self.emb_encoders[0](inputs[0][0][0])
            rnn_input = torch.cat(embs, dim=2)
            # Similarity score softmax to use in the current measure.
            # This is not used for the first measure.

            if train_mode:
                # Train mode. Need to save this as a list because of generate-mode.
                prev_data = {'score_softmax': [None for b in range(bsz)], 'encs': [[] for b in range(bsz)]}
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                # Note that t is input-indexed
                if curr_t is not None: t = curr_t
                # We initialize the score_softmax to be such that the model prefers 
                # the backbone RNN in the first measure. TODO magic number
                # Encoding is based on input indexing, not outputs.
                for b in range(bsz):
                    emb_t_b = emb_t.squeeze(1)[b].unsqueeze(0)
                    # In generate mode, t=0 even though we're at t=t, so index |inputs| with 0
                    t_to_use = t if train_mode else 0
                    emb_t_b_enc = self.prev_emb_encoders[0](inputs[0][b][t_to_use]) 
                    emb_t_b_dec = self.prev_emb_encoders_2[0](inputs[0][b][t_to_use]) 

                    # Run the "encoder" RNN
                    hidden['prev_enc'][b] = \
                            self.prev_enc_rnn(
                                    emb_t_b_enc, hidden['prev_enc'][b])

                    if t in beg_idxs[b] and t != 0:
                        # Add the prev measure encoding to prev_data
                        prev_data['encs'][b].append(hidden['prev_enc'][b])

                        # Sets |score_softmax| to the score softmax that'll be
                        # used in the gating at _this_ measure. 
                        # print conditions[b][len(prev_data[b])]
                        decoder_h0, score_softmax = self.self_attention(
                                hidden['backbone'][b],
                                prev_data['encs'][b], 
                                conditions[b][len(prev_data['encs'][b])],
                                args)
                        prev_data['score_softmax'][b] = score_softmax

                        # Init the decoder RNN
                        hidden['prev_dec'][b] = decoder_h0

                        # Reset prev_enc RNN
                        hidden['prev_enc'][b] = Variable(weight.new(1, self.nhid).zero_())


                    # Run the decoder RNN
                    hidden['prev_dec'][b] = \
                            self.prev_dec_rnn(emb_t_b_dec, hidden['prev_dec'][b])

                # Run the backbone RNN
                hidden['backbone'] = self.rnn(emb_t.squeeze(1), hidden['backbone'])

                # Replace output with decoder output if we're decoding a measure 
                to_concat = []
                for b in range(bsz):
                    h_backbone = hidden['backbone'][b].unsqueeze(0)
                    h_prev = hidden['prev_dec'][b]
                    if len(beg_idxs[b]) <= 1 or t < beg_idxs[b][1]:
                        # Do not use decoder in first measure. beg_idxs[b][0] is 0.
                        to_concat.append(h_backbone)
                    else:
                        to_concat.append(self.get_new_output(
                            h_backbone, h_prev, prev_data['score_softmax'][b], args))

                output += [torch.cat(to_concat, dim=0)]

            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.emb_decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))

            return decs, hidden

# End MRNN


class ERNNModel(nn.Module):
    def __init__(self, args):
        # Note: emsize of prev_emb is just nhid for now.
        super(ERNNModel, self).__init__()
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        ntokens = args.ntokens
         
        self.num_channels = len(ntokens)
        self.drop = nn.Dropout(dropout)
        self.nhid = nhid # output dimension
        for i in range(self.num_channels):
            self.add_module('emb_encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
            self.add_module('prev_emb_encoder_' + str(i), nn.Embedding(ntokens[i], self.nhid)) 
        self.rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        for i in range(len(ntokens)):
            self.add_module('emb_decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        self.emb_encoders = AttrProxy(self, 'emb_encoder_') 
        self.prev_emb_encoders = AttrProxy(self, 'prev_emb_encoder_') 
        self.emb_decoders = AttrProxy(self, 'emb_decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nlayers = nlayers
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.A = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid).zero_())
        nn.init.xavier_normal(self.A)
        self.B = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid).zero_())
        nn.init.xavier_normal(self.B)
        self.default_enc = nn.Parameter(torch.FloatTensor(self.nhid).zero_())
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.emb_encoders[i].weight.data)
            nn.init.xavier_normal(self.prev_emb_encoders[i].weight.data)
            nn.init.xavier_normal(self.emb_decoders[i].weight.data)
            self.emb_decoders[i].bias.data.fill_(0)

    def is_lstm(self):
        return self.rnn_type == 'LSTM'

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())
   
    def get_new_output(self, hidden, prev_data, conditions, batch_size, t, args):
        to_concat = []
        for b in range(batch_size):
            # Sometimes, t is the length of conditions minus 1, because we could be
            # be trying to get the next condition, but we're already at the end.
            if t == len(conditions[b])-1:
                prev_idx = -1
            else:
                prev_idx = conditions[b][t+1]
            use_prev = t > args.skip_first_n_note_losses and prev_idx != -1 
            prev = prev_data[prev_idx][b] if use_prev else hidden[b]
            to_concat.append(prev.unsqueeze(0))
        return torch.cat(to_concat, dim=0)

    def encode_batch_t(self, inputs, t): 
        batch_size = inputs[0].size(0)
        to_concat = [] 
        for b in range(batch_size):
            to_concat.append(self.prev_emb_encoders[0](inputs[0][b][t]))
        return torch.cat(to_concat, dim=0)

    # TODO implement this for ERNN
    def forward_ss(self, data, hidden, args, prev_data=None):
         return 

    def forward(self, data, hidden, args, prev_data=None, curr_t=None):
        # hidden is a dict
        # what we call "prev_data" is not actually previous hidden states.
        train_mode = prev_data is None
        if args.ss:
            return self.forward_ss(data, hidden, args, prev_data)
        else:
            # list of (bsz,seqlen)
            inputs = data["data"]
            # data["conditions"] is a list of (bsz,seqlen)
            conditions = data["conditions"][0].data.tolist() 
           
            # print inputs[0][0], conditions[0]
            # print "-"*88
            output = []
            batch_size = inputs[0].size(0)
            embs = []
            for c in range(self.num_channels):
                embs.append(self.drop(self.emb_encoders[c](inputs[c])))
            rnn_input = torch.cat(embs, dim=2)
            if train_mode:
                # Train mode. Need to save this as a list because of generate-mode.
                prev_data = []
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                # Encoding is based on input indexing, not outputs.
                # In generate mode, this works by always appending t=0, which is at t=t
                prev_data.append(self.encode_batch_t(inputs, t))
                if not train_mode: t = curr_t
                hidden = self.rnn(emb_t.squeeze(1), hidden)
                output += [self.get_new_output(
                    hidden, prev_data, conditions, batch_size, t, args)]
            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.emb_decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))

            """
            # for bsz=1
            if train_mode:
                for t in range(decs[0].size(1)):
                    if t == decs[0].size(1)-1:
                        prev_idx = -1
                    else:
                        prev_idx = conditions[0][t+1]
                    w = 0
                    use_prev = t > args.skip_first_n_note_losses and prev_idx != -1 and prev_idx < t-w
                    out = prev_data[prev_idx]
                    if use_prev:
                        # print prev_data, t, prev_idx+w, out
                        for i in range(41):
                            decs[0][0,t,i].data.zero_()
                        decs[0][0,t,out].data.add_(1)
            """

            return decs, hidden



## END ERNN




"""
UNUSED MODELS
"""

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
            self.add_module('emb_encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        self.rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        self.parallel_rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)
        for i in range(len(ntokens)):
            self.add_module('emb_decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        self.emb_encoders = AttrProxy(self, 'emb_encoder_') 
        self.emb_decoders = AttrProxy(self, 'emb_decoder_') 

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
            nn.init.xavier_normal(self.emb_encoders[i].weight.data)
            nn.init.xavier_normal(self.emb_decoders[i].weight.data)
            self.emb_decoders[i].bias.data.fill_(0)

    def is_lstm(self):
        return self.rnn_type == 'LSTM'

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.is_lstm():
            main_hidden= (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
            parallel_hidden= (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            main_hidden = Variable(weight.new(bsz, self.nhid).zero_())
            parallel_hidden = Variable(weight.new(bsz, self.nhid).zero_())
        return {'main': main_hidden, 'parallel': parallel_hidden}
    
    def get_new_h_t(self, curr_h, prev_data, conditions, batch_size, t, args):
        to_concat = []
        for b in range(batch_size):
            if args.baseline:
                prev = self.default_h
            else:
                prev_idx = conditions[b][t]
                w = 1
                use_prev = t > args.skip_first_n_note_losses and prev_idx != -1 and prev_idx < t-w
                prev = prev_data[prev_idx+w][b] if use_prev else self.default_h
            l = torch.matmul(self.A, curr_h[b])
            r = torch.matmul(self.B, prev)
            to_concat.append(l.unsqueeze(0)+r.unsqueeze(0))
        return torch.cat(to_concat, dim=0)


    # TODO implement this for PRNN
    def forward_ss(self, data, hidden, args, prev_data=None):
         return 

    def forward(self, data, hidden, args, prev_data=None, curr_t=None):
        # hidden is a dict
        if args.ss:
            return self.forward_ss(data, hidden, args, prev_data)
        else:
            # list of (bsz,seqlen)
            inputs = data["data"]
            train_mode = prev_data is None
            # data["conditions"] is a list of (bsz,seqlen)
            conditions = data["conditions"][0].data.tolist() 
            output = []
            batch_size = inputs[0].size(0)
            embs = []
            for c in range(self.num_channels):
                embs.append(self.drop(self.emb_encoders[c](inputs[c])))
            rnn_input = torch.cat(embs, dim=2)
            if prev_data is None:
                # Train mode
                prev_data = [hidden["parallel"][0] if self.is_lstm() else hidden["parallel"]]
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                main_h = hidden["main"]
                parallel_h = hidden["parallel"]
                curr_main_h = main_h[0] if self.is_lstm() else main_h
                new_h_t = self.get_new_h_t(curr_main_h, prev_data, conditions, batch_size, t, args)
                if self.is_lstm():
                    hidden["main"] = self.rnn(emb_t.squeeze(1), (new_h_t, main_h[1]))
                    hidden["parallel"] = self.parallel_rnn(emb_t.squeeze(1), parallel_h)
                else:
                    hidden["main"] = self.rnn(emb_t.squeeze(1), new_h_t)
                    hidden["parallel"] = self.rnn(emb_t.squeeze(1), parallel_h)
                prev_data.append(parallel_h[0] if self.is_lstm() else parallel_h)
                output += [hidden["main"][0] if self.is_lstm() else hidden["main"]]
            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.emb_decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
            return decs, hidden


## END PRNN


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
            self.add_module('emb_encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
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
            self.add_module('emb_decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        self.emb_encoders = AttrProxy(self, 'emb_encoder_') 
        self.emb_decoders = AttrProxy(self, 'emb_decoder_') 

        self.init_weights()

        self.rnn_type = args.rnn_type
        self.nlayers = nlayers

    def init_weights(self):
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.emb_encoders[i].weight.data)
            nn.init.xavier_normal(self.emb_decoders[i].weight.data)
            self.emb_decoders[i].bias.data.fill_(0)

    def forward(self, data, hidden, prev_data=None):
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
            embs.append(self.drop(self.emb_encoders[c](inputs[c])))
        rnn_input = torch.cat(embs, dim=2)
        if prev_data is None:
            prev_data = [hidden]
        for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
            to_cat = []
            for b in range(batch_size):
                # TODO make this work for LSTMs
                to_cat.append(torch.unsqueeze(prev_data[conditions[b][t]][b], 0))
            new_h = torch.cat(to_cat, dim=0)
            hidden = self.rnn(emb_t.squeeze(1), new_h)
            prev_data.append(hidden)
            output += [hidden[0] if self.rnn_type == 'LSTM' else hidden]
        output = torch.stack(output, 1)
        output = self.drop(output)

        decs = []
        for i in range(self.num_channels):
            decoded = self.emb_decoders[i](
                output.view(output.size(0)*output.size(1), output.size(2)))
            decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
        return decs, prev_data[-1]

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())


## END VINE


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
            self.add_module('emb_encoder_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
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
            self.add_module('emb_decoder_' + str(i), nn.Linear(self.nhid, ntokens[i])) 
        self.emb_encoders = AttrProxy(self, 'emb_encoder_') 
        self.emb_decoders = AttrProxy(self, 'emb_decoder_') 

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
            nn.init.xavier_normal(self.emb_encoders[i].weight.data)
            nn.init.xavier_normal(self.emb_decoders[i].weight.data)
            self.emb_decoders[i].bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, 2*self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())

    def get_new_h_t(self, prev_data, conditions, batch_size, t, args):
        to_concat = []
        for b in range(batch_size):
            prev_idx = conditions[b][t]
            # prev = prev_data[prev_idx+1][b] if t > args.skip_first_n_note_losses and prev_idx != -1 and prev_idx != t-1 else self.default_h
            prev = prev_data[prev_idx][b] if t > args.skip_first_n_note_losses and prev_idx != -1 else self.default_h
            # prev = self.default_h
            l = torch.matmul(self.A, prev_data[-1][b])
            r = torch.matmul(self.B, prev)
            to_concat.append(l.unsqueeze(0)+r.unsqueeze(0))
        return torch.cat(to_concat, dim=0)


    def forward_ss(self, data, hidden, args, prev_data=None):
        # linear annealing 
        prob_gold = max(float(args.epochs-args.epoch)/args.epochs, 0)

        # list of (bsz,seqlen)
        inputs = data["data"]
        # data["conditions"] is a list of (bsz,seqlen)
        conditions = data["conditions"][0].data.tolist() # TODO
        batch_size = inputs[0].size(0)
        if prev_data is None:
            # Train mode
            prev_data = [hidden[0] if self.rnn_type == 'LSTM' else hidden]
                 
        decs = [[]*self.num_channels]
        tmp = []
        for c in range(self.num_channels):
            tmp.append(self.drop(self.emb_encoders[c](inputs[c][:,0])))
        emb_t = torch.cat(tmp, dim=1)
        for t in range(inputs[0].size(1)):
            new_h_t = self.get_new_h_t(prev_data, conditions, batch_size, t, args)
            if self.rnn_type == 'LSTM': 
                hidden = self.rnn(emb_t.squeeze(1), (new_h_t, hidden[1]))
            else:
                hidden = self.rnn(emb_t.squeeze(1), new_h_t)
            out_t = hidden[0] if self.rnn_type == 'LSTM' else hidden
            prev_data.append(out_t)
            out_t = self.drop(out_t)
            tmp = []
            for c in range(self.num_channels):
                decoded = self.emb_decoders[c](out_t)
                decs[c].append(decoded.unsqueeze(1))
                weights = torch.stack([F.softmax(decoded.data.div(args.temperature)[i]) for i in range(batch_size)], 0) 
                sampled_idxs = torch.multinomial(weights, 1)
                idx = inputs[c][:,t] if random.random() < prob_gold else sampled_idxs.squeeze()
                tmp.append(self.drop(self.emb_encoders[c](idx)))
            emb_t = torch.cat(tmp, dim=1)

        for c in range(self.num_channels):
            decs[c] = torch.cat(decs[c], dim=1)

        return decs, hidden


    def forward(self, data, hidden, args, prev_data=None):
        ''' input should be a list with aligned inputs for each channel '''
        ''' conditions should be a list with indices of the prev hidden states '''
        ''' for conditioning. -1 means no condition '''
        ''' returns a list of outputs for each channel '''
        # print self.default_h
        if args.ss:
            return self.forward_ss(data, hidden, args, prev_data)
        else:
            # list of (bsz,seqlen)
            inputs = data["data"]
            # data["conditions"] is a list of (bsz,seqlen)
            conditions = data["conditions"][0].data.tolist() # TODO
            output = []
            batch_size = inputs[0].size(0)
            embs = []
            for c in range(self.num_channels):
                embs.append(self.drop(self.emb_encoders[c](inputs[c])))
            rnn_input = torch.cat(embs, dim=2)
            if prev_data is None:
                # Train mode
                prev_data = [hidden[0] if self.rnn_type == 'LSTM' else hidden]
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                new_h_t = self.get_new_h_t(prev_data, conditions, batch_size, t, args)
                if self.rnn_type == 'LSTM': 
                    hidden = self.rnn(emb_t.squeeze(1), (new_h_t, hidden[1]))
                else:
                    hidden = self.rnn(emb_t.squeeze(1), new_h_t)
                prev_data.append(hidden[0] if self.rnn_type == 'LSTM' else hidden)
                output += [prev_data[-1]]
            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.emb_decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
            return decs, prev_data[-1]

# END XRNN

