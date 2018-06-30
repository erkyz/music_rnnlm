import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import random
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class RNNCellModel(nn.Module):
    ''' 
    Basic RNN model that uses nn.RNNCell
    This module is primarily intended to be a parent class to build other models off of.
    You should use rnnlm.RNNModel if you want to train a baseline RNN, since unrolling 
    using nn.RNNCell is slower than using nn.RNN.
    '''
    def __init__(self, args):
        '''
        |args| is a argparse object, provided by main.py. This provides any flags
            and settings for this training run.
        '''

        super(RNNCellModel, self).__init__()
        # A list of the number of tokens in each channel's vocab
        ntokens = args.ntokens 

        self.num_channels = len(args.ntokens)
        self.drop = nn.Dropout(args.dropout)
        self.rnn_type = args.rnn_type
        self.nhid = args.nhid
        self.nlayers = args.nlayers
        self.dtype = args.cuda

        for i in range(self.num_channels):
            self.add_module('backbone_embedding_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
        self.rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, args.nhid, args.nlayers)
        for i in range(self.num_channels):
            self.add_module('emb_decoder_' + str(i), nn.Linear(args.nhid, ntokens[i])) 

        # Embeddings for the main RNN, called the "backbone RNN." In this case, it is
        # the only RNN in the model. Access like a list, where index is the channel idx.
        self.backbone_embeddings = AttrProxy(self, 'backbone_embedding_')

        # Decoder for the embedding back to the original vocab size.
        self.emb_decoders = AttrProxy(self, 'emb_decoder_') 

        self.init_weights(args)

    def init_weights(self, args):
        # Xavier initialization
        for c in range(self.num_channels):
            nn.init.xavier_normal(self.backbone_embeddings[c].weight.data)
            nn.init.xavier_normal(self.emb_decoders[c].weight.data)
            self.emb_decoders[c].bias.data.fill_(0)

    @property
    def need_conditions(self):
        return False

    # Scheduled sampler
    # NOTE I haven't verified whether this works in a while.
    def forward_ss(self, data, hidden, args):
        inputs = data["data"]
        batch_size = inputs[0].size(0)
        # Linear annealing 
        prob_gold = max(float(args.epochs-args.epoch)/args.epochs, 0)
        decs = [[]*self.num_channels]

        tmp = []
        for c in range(self.num_channels):
            tmp.append(self.drop(self.backbone_embeddings[c](inputs[c][:,0])))
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
                tmp.append(self.drop(self.backbone_embeddings[c](idxs)))
            emb_t = torch.cat(tmp, dim=1)

        for c in range(self.num_channels):
            decs[c] = torch.cat(decs[c], dim=1)

        return decs, hidden

    def forward(self, data, hidden, args, prev_data=None, curr_t=None):
        '''
        |data| is a dict that can include things like the conditions. For the basic 
            RNNCellModel, it just has a "data" field.
        |hidden| is the hidden state. It is a tuple if rnn_type is LSTM.
            NOTE |hidden| should be initialized using the init_hidden function.
        |args| is a argparse object, provided by main.py. This provides any flags
            and settings for this training run.
        |prev_data| and |curr_t| are not used by RNNCellModel.
        '''
        if args.ss:
            return self.forward_ss(data, hidden, args)
        else:
            inputs = data["data"]
            output = []
            embs = []
            batch_size = inputs[0].size(0)
            for i in range(self.num_channels):
                embs.append(self.drop(self.backbone_embeddings[i](inputs[i])))
            rnn_input = torch.cat(embs, dim=2)
            # TODO use rnn.pack_padded_sequence to save computation?
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
        # Zero initialization
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())


class AttentionRNNModel(RNNCellModel):
    '''
    Model that implements simple multiplicative attention (Luong attention, from
        "Effective approaches to attention-based neural machine translation")
    '''

    def __init__(self, args):
        super(AttentionRNNModel, self).__init__(args)

    def init_weights(self, args):
        self.W_a = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid).zero_())
        nn.init.xavier_normal(self.W_a)
        self.W_c = nn.Parameter(torch.FloatTensor(self.nhid,self.nhid*2).zero_())
        nn.init.xavier_normal(self.W_c)
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.backbone_embeddings[i].weight.data)
            nn.init.xavier_normal(self.emb_decoders[i].weight.data)
            self.emb_decoders[i].bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        backbone = Variable(weight.new(bsz, self.nhid).zero_())
        return backbone
   
    def get_self_attention_new_h(self, h_t, prev_data, args):
        '''
        Luong self-attention over previous segment encodings
        |h_t| is the current hidden state
        |prev_data| should be a list of previous hidden states. This makes this 
            function VERY slow for long examples.
        '''

        vs = []
        for h_i in prev_data:
            # Scoring fucntion between the current hidden state (h_t) and
            # the previous hidden state at step i.
            # Use batched matrix-vector product (BMV)
            vs.append(torch.matmul(
                h_t.unsqueeze(1), 
                torch.matmul(self.W_a.unsqueeze(0), h_i.unsqueeze(2))
            ).squeeze(2))

        # Softmax the scores.
        scores = util.softmax2d(torch.cat(vs, dim=1)).permute(1,0) # t x bsz
        bsz = scores.size(1)
        # Get context vector
        c_t = sum([torch.mul(scores[t].unsqueeze(1).expand(bsz, args.nhid),
                    prev_data[t]) for t in range(len(prev_data))])
        # Batched matrix-vector product (BMV) to get new attentional hidden state
        # for timestep t.
        x = torch.matmul(
                self.W_c.unsqueeze(0), 
                torch.cat([h_t, c_t], dim=1).unsqueeze(2)
             ).squeeze(2)
        return F.tanh(x)

    def forward_ss(self, data, hidden, args, prev_data=None):
        # TODO implement scheduled sampling for AttentionRNNModel
         return 

    def forward(self, data, hidden, args, prev_data=None, curr_t=None):
        '''
        See forward method of parent class for more info. 
        TODO: AttentionRNNModel has not been implemented to support rnn_type=LSTM. Should
            be an easy change.
        For AttentionRNNModel, |prev_data| should be a list of previous hidden states.
            Need to save this as a list because it needs to be passed step-by-step in
            generate-mode.
        NOTE that we can also use a fixed attention window ("local attention"), which is 
            not implemented but would not require a large change in the code.
        '''
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
                embs.append(self.drop(self.backbone_embeddings[c](inputs[c])))
            emb_start = self.backbone_embeddings[0](inputs[0][0][0])
            rnn_input = torch.cat(embs, dim=2)

            if train_mode:
                # Train mode. 
                prev_data = []
            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                # Note that t is input-indexed
                # In generate mode, t=0 even though we're at t=t, so index |inputs| with 0
                t_to_use = t if train_mode else 0
            
                # Run RNN
                h_t = self.rnn(emb_t.squeeze(1), hidden)
                # From the paper, it looks like we still continue passing h_t as the 
                # hidden state, and only use h_t_tilde for calculating the output.
                prev_data.append(h_t)
                hidden = h_t
                # Get attentional vector h_t_tilde
                h_t_tilde = self.get_self_attention_new_h(h_t, prev_data, args)
                output += [h_t_tilde]

            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.emb_decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))
            return decs, hidden

# END AttentionRNNModel


class READRNNModel(RNNCellModel):
    def __init__(self, args):
        super(READRNNModel, self).__init__(args)
        # NOTE that we can generalize anything that says "measure" to arbitrary-length 
        # segments of a song.
        
        for i in range(self.num_channels):
            self.add_module('encoder_rnn_embedding_' + str(i), nn.Embedding(ntokens[i], args.emsize)) 
            self.add_module('decoder_rnn_embedding_' + str(i), nn.Embedding(ntokens[i], args.emsize))
        insize = args.emsize+args.input_feed_dim if args.input_feed_num else args.emsize

        # "Encoder RNN"
        self.encoder_rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)

        # "Decoder RNN"
        self.decoder_rnn = getattr(nn, args.rnn_type + 'Cell')(
            args.emsize*self.num_channels, self.nhid, nlayers)

        # For the scoring function
        self.fc1 = nn.Linear(self.nhid+1, self.gates_nhid) # +1 for the simscore
        self.fc2 = nn.Linear(self.gates_nhid, 1)
        # For output gate
        self.fc3 = nn.Linear(1, self.gates_nhid)
        self.fc4 = nn.Linear(self.gates_nhid, 1)
        # For input feeding
        self.fc5 = nn.Linear(args.input_feed_num+1, self.gates_nhid) # +1 for num_left
        self.fc6 = nn.Linear(self.gates_nhid, args.input_feed_dim)
        # For encoder RNN
        self.prev_backbone_embeddings = AttrProxy(self, 'encoder_rnn_embedding_') 
        # For decoder RNN
        self.decoder_rnn_embeddings = AttrProxy(self, 'decoder_rnn_embedding_') 

    def init_weights(self, args):
        self.default_future = nn.Parameter(torch.FloatTensor(args.input_feed_num).zero_())
        if args.cuda:
            self.default_future.cuda()
        for i in range(self.num_channels):
            nn.init.xavier_normal(self.backbone_embeddings[i].weight.data)
            nn.init.xavier_normal(self.prev_backbone_embeddings[i].weight.data)
            nn.init.xavier_normal(self.decoder_rnn_embeddings[i].weight.data)
            nn.init.xavier_normal(self.emb_decoders[i].weight.data)
            self.emb_decoders[i].bias.data.fill_(0)

    @property
    def need_conditions(self):
        return False

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        backbone = Variable(weight.new(bsz, self.nhid).zero_())
        enc_rnn = [Variable(weight.new(1, self.nhid).zero_()) for b in range(bsz)]
        dec_rnn = [Variable(weight.new(1, self.nhid).zero_()) for b in range(bsz)]
        return {'backbone': backbone, 'enc_rnn': enc_rnn, 'dec_rnn': dec_rnn}
   
    def output_gate(self, h_backbone, h_dec, delta_tilde, args):
        # Weighted average of |h_backbone| and |h_dec|, based on |delta_tilde|
        x = delta_tilde
        x = self.fc3(x) 
        x = self.fc4(x)
        alpha = F.sigmoid(x)
        '''
        if random.random() < 0.01:
            print delta_tilde.data[0], alpha.data[0]
        '''
        return alpha*h_dec + (1-alpha)*h_backbone

    def get_future_encoding(self, ssm, curr_measure, num_left, args):
        # Get a mapping of the next |args.input_feed_num| measure edit distances, 
        # plus the number of notes to end of measure
        if curr_measure >= len(ssm[curr_measure]) - args.input_feed_num:
            next_scores = self.default_future
            num_left_tensor = Variable(
                    torch.FloatTensor([num_left]), requires_grad=False).type(self.dtype)
            s = torch.cat([next_scores, num_left_tensor])
        else:
            next_scores = ssm[curr_measure][curr_measure+1:curr_measure+args.input_feed_num+1]
            s = Variable(torch.FloatTensor(
                np.append(next_scores,num_left)), requires_grad=False).type(self.dtype)

        # Run fully-connected layer
        x = self.fc5(s)
        x = F.tanh(self.fc6(x))
        return x

    def self_attention(self, h_backbone, prev_measure_encs, delta, args):
        '''
        "Self-attention" over previous segment encodings.
        Using paper terminology, this returns 
        1) |decoder_h0|, the weighted average of previous measure encodings, weighted
            by the scores of the previous measures
            Note: |decoder_h0| named "e tilde" in paper
        2) |delta_tilde|, the average distance between the current measure and the 
            previous measures, weighted by the scores of the previous measures.

        Arguments:
        |h_backbone| is the current backbone hidden state
            NOTE not currently used, but could [should] be used in the scoring function
        |prev_measure_encs| are the previous encodings of previous measures
        |delta|: the ith element of |delta| is the distance (given in the SDM) between 
            the current measure and the ith previous measure
        '''

        # |vs| will be a list of the previous measure scores, scored below
        vs = []
        for i, enc_rnn in enumerate(prev_measure_encs):
            # Scoring function of the ith previous measure 
            s = Variable(torch.FloatTensor([delta[i]]), requires_grad=False).type(self.dtype)
            x = torch.cat([enc_rnn.squeeze(), s])
            x = self.fc1(x) 
            x = F.tanh(self.fc2(x))
            vs.append(x)
        softmax = F.softmax(torch.cat(vs))
        decoder_h0 = torch.sum(
                torch.cat(
                    [prev_measure_encs[i]*softmax[i] for i in range(len(prev_measure_encs))]),
                0).unsqueeze(0)
        delta_tilde = torch.sum(torch.cat(
            [torch.mul(
                torch.FloatTensor([delta[i]]), requires_grad=False).type(self.dtype),
                softmax[i]
                ) for i in range(len(prev_measure_encs))]
            ), 0)
        return decoder_h0, delta_tilde

    def forward_ss(self, data, hidden, args, prev_data=None):
        # TODO implement scheduled sampling for READ-RNN
         return 

    def forward(self, data, hidden, args, prev_data=None, curr_t=None):
        '''
        |data| must have "data", "conditions", and "metadata" fields.
        |hidden| should be initialized using the init_hidden function.
            Here, |hidden| is a dict, unlike in the forward function of RNNCellModel.
            TODO: READ-RNN has not been implemented to support rnn_type=LSTM, since the hidden
            state is a tuple for rnn_type=LSTM. To modify this class, you would need to 
            pass the LSTM hidden state (and not the cell state) around.
        |prev_data| is a dict, in this case with two fields:
            1) "delta_tilde", the delta_tilde value at this measure
            2) "encs", the previous measure encodings
            This is used because in generate-mode, we generate note-by-note and need to
            pass in the previous information manually.
        |curr_t| is the current timestep, used in generate mode, again since we must keep
            track of where we are. In train-mode, curr_t should not be passed in.
        '''

        train_mode = curr_t is None
        if args.ss:
            return self.forward_ss(data, hidden, args, prev_data)
        else:
            # list of (bsz,seqlen)
            inputs = data["data"]
            bsz = inputs[0].size(0)

            # For READ-RNN the SDM is passed in through |conditions|
            sdm = data["conditions"][0]

            # For READ-RNN the measure boundary indices are passed in as "metadata"
            # |measure_boundaries| is indexed ignoring the START token, 
            # so it's aligned with the outputs!
            measure_boundaries = data["metadata"][0]

            # Indices of beginnings of all measures
            beg_idxs = [[s[0] for s in measure_boundaries[b]] for b in range(bsz)]

            # Indices of the beginnings of upcoming measures from our timestep
            future_beg_idxs = [[s[1] for s in measure_boundaries[b]] for b in range(bsz)]
           
            weight = next(self.parameters()).data
            output = []

            # Embed inputs
            embs = []
            for c in range(self.num_channels):
                embs.append(self.drop(self.backbone_embeddings[c](inputs[c])))
            emb_start = self.backbone_embeddings[0](inputs[0][0][0])
            rnn_input = torch.cat(embs, dim=2)

            if train_mode:
                # Train mode, create prev_data from scratch.
                # Need to save this as a list because of generate-mode.
                prev_data = {
                        'delta_tilde': [None for b in range(bsz)], 
                        'encs': [[] for b in range(bsz)]
                        }

            for t, emb_t in enumerate(rnn_input.chunk(rnn_input.size(1), dim=1)):
                # NOTE Encoding is based on input indexing, not outputs.
                # Note that t is input-indexed
                if curr_t is not None: t = curr_t

                # Split the batch into individual examples, since we have to attend 
                # differently for each batch example.
                for b in range(bsz):
                    emb_t_b = emb_t.squeeze(1)[b].unsqueeze(0)
                    # In generate mode, variable t is 0 even though we're not at time 0.
                    # When generating, we provide only this step input, so set t=0
                    t_to_use = t if train_mode else 0
                    emb_t_b_enc = self.prev_backbone_embeddings[0](inputs[0][b][t_to_use]) 
                    emb_t_b_dec = self.decoder_rnn_embeddings[0](inputs[0][b][t_to_use]) 

                    # Run the encoder RNN first
                    hidden['enc_rnn'][b] = \
                            self.encoder_rnn(
                                    emb_t_b_enc, hidden['enc_rnn'][b])

                    if t in beg_idxs[b] and t != 0:
                        # We're at the beginning of a measure (other than the first measure)

                        # Update future_beg_idxs
                        future_beg_idxs[b] = future_beg_idxs[b][1:]

                        # Add the previous measure's encoding to prev_data
                        prev_data['encs'][b].append(hidden['enc_rnn'][b])

                        decoder_h0, delta_tilde = self.self_attention(
                                h_backbbone=hidden['backbone'][b],
                                prev_measure_encs=prev_data['encs'][b], 
                                delta=sdm[b][len(prev_data['encs'][b])],
                                args=args)
                        prev_data['delta_tilde'][b] = delta_tilde

                        # Init the decoder RNN
                        hidden['dec_rnn'][b] = decoder_h0

                        # Reset encoder RNN, since we're at a new measure
                        hidden['enc_rnn'][b] = Variable(weight.new(1, self.nhid).zero_())


                    # Run the decoder RNN 
                    hidden['dec_rnn'][b] = \
                            self.decoder_rnn(emb_t_b_dec, hidden['dec_rnn'][b])

                # Run the backbone RNN
                if args.input_feed_num > 0:
                    # For input feeding, we encode and then feed in the SDM values 
                    # for the next |args.input_feed_num| measures. We then encode this
                    # to the input to the backbone RNN.
                    # We also feed the number of notes to the end of the measure here.
                    # NOTE feeding future deltas hasn't worked so far using this 
                    # implementation
                    # TODO feed the number of notes left in the measure separately.
                    future_by_batch = []
                    for b in range(bsz):
                        curr_measure = len(prev_data['encs'][b])
                        num_left = future_beg_idxs[b][0] - t
                        future_by_batch.append(self.get_future_encoding(
                            sdm[b], curr_measure, num_left, args
                            ).unsqueeze(0))
                    inp_t = torch.cat([emb_t.squeeze(1),
                        torch.cat(future_by_batch, dim=0)], dim=1)
                    hidden['backbone'] = self.rnn(inp_t, hidden['backbone'])
                else:
                    hidden['backbone'] = self.rnn(emb_t.squeeze(1), hidden['backbone'])

                batch_output = []
                for b in range(bsz):
                    h_backbone = hidden['backbone'][b].unsqueeze(0)
                    h_prev = hidden['dec_rnn'][b]
                    if len(beg_idxs[b]) <= 1 or t < beg_idxs[b][1]:
                        # Don't use the decoder on the first measure.
                        # NOTE The model seems to converge better if you initialize
                        # |delta_tilde| to 10 in the first measure, which is a magic
                        # number that's higher than any distance. But that's hacky.
                        batch_output.append(h_backbone)
                    else:
                        # Use the output gate to determine the mixture of backbone RNN
                        # and decoder RNN outputs
                        batch_output.append(self.output_gate(
                            h_backbone, h_prev, prev_data['delta_tilde'][b], args))

                output += [torch.cat(batch_output, dim=0)]

            output = torch.stack(output, 1)
            output = self.drop(output)

            decs = []
            for i in range(self.num_channels):
                decoded = self.emb_decoders[i](
                    output.view(output.size(0)*output.size(1), output.size(2)))
                decs.append(decoded.view(output.size(0), output.size(1), decoded.size(1)))

            return decs, hidden

# End READRNN
