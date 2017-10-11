

import dynet as dy
import time, util

import rnnlm

class BaselineDyGenRNNLM(rnnlm.SaveableRNNLM):
    name = "baseline_dynet"

    def add_params(self):
        if self.args.rnn == "lstm": rnn = dy.LSTMBuilder
        elif self.args.rnn == "gru": rnn = dy.GRUBuilder
        else: rnn = dy.SimpleRNNBuilder

        # GENERATIVE MODEL PARAMETERS
        self.lookup = self.model.add_lookup_parameters((self.vocab.size, self.args.input_dim))
        self.rnn = rnn(self.args.layers, self.args.input_dim, self.args.hidden_dim, self.model)
        self.R = self.model.add_parameters((self.vocab.size, self.args.hidden_dim))
        self.bias = self.model.add_parameters((self.vocab.size,))

        # print self.vocab.size, self.args.hidden_dim, self.args.input_dim

    def BuildLMGraph(self, mel, mel_args=None):
        dy.renew_cg()

        APPLY_DROPOUT = self.args.dropout is not None and ("test" not in mel_args or mel_args["test"] != True)
        if APPLY_DROPOUT: self.rnn.set_dropout(self.args.dropout)
        else: self.rnn.disable_dropout()

        # GENERATIVE MODEL
        s0 = self.rnn.initial_state()
        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)
        errs = [] # will hold expressions
        state = s0
        for (cm,nm) in zip(mel,mel[1:]):
            x_t = self.lookup[cm]
            state = state.add_input(x_t)
            y_t = state.output()
            # if APPLY_DROPOUT: y_t = dy.dropout(y_t, self.args.dropout)
            r_t = bias + (R * y_t)
            err = dy.pickneglogsoftmax(r_t, int(nm))
            errs.append(err)
        err = dy.esum(errs)
        return err


    # note: batch is an ndarray of tuples (pitch, duration) in my current implementation of Vocab

    def BuildLMGraph_batch(self, batch, mel_args=None):
        # if "skip_renew" not in mel_args: 
        dy.renew_cg()

        # TODO
        APPLY_DROPOUT = self.args.dropout is not None # and ("test" not in mel_args or mel_args["test"] != True)
        if APPLY_DROPOUT: self.rnn.set_dropout(self.args.dropout)
        else: self.rnn.disable_dropout()

        s0 = self.rnn.initial_state()

        #MASK MELODIES
        mels_padded = [] # Dimension: max_melody_len * minibatch_size
        masks = [] # Dimension: max_melody_len * minibatch_size

        #No of words processed in this batch
        max_melody_len = max([len(mel) for mel in batch])

        # create padded batches
        for mel in batch:
            mels_padded.append([self.vocab[e].i for e in mel] + [self.vocab[self.vocab.END_EVENT].i for _ in range(max_melody_len-len(mel))])
            masks.append( [1 for _ in mel] + [0 for _ in range(max_melody_len-len(mel))])
        mels_padded = map(list, zip(*mels_padded)) # transposes
        masks = map(list, zip(*masks))

        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)
        errs = [] # will hold expressions
        state = s0

        t = time.time()
        for (mask, curr_words, next_words) in zip(masks[1:], mels_padded, mels_padded[1:]):
            x_t = dy.lookup_batch(self.lookup, curr_words)
            state = state.add_input(x_t)
            y_t = state.output()
            if APPLY_DROPOUT: y_t = dy.dropout(y_t, self.args.dropout)
            r_t = bias + (R * y_t)
            err = dy.pickneglogsoftmax_batch(r_t, next_words)

            mask_expr = dy.inputVector(mask)
            mask_expr = dy.reshape(mask_expr, (1,), len(mask))
            err = err * mask_expr
            errs.append(err)

            t = time.time()
        nerr = dy.esum(errs)
        return nerr

    def sample(self, mel_args={}, max_events=200):
        dy.renew_cg()

        APPLY_DROPOUT = self.args.dropout is not None and ("test" not in mel_args or mel_args["test"] != True)
        if APPLY_DROPOUT: self.rnn.set_dropout(self.args.dropout)
        else: self.rnn.disable_dropout()

        # GENERATIVE MODEL
        s0 = self.rnn.initial_state()
        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)
        errs = [] # will hold expressions
        events = []
        state = s0
        cm = self.vocab[self.vocab.START_EVENT].i
        while len(events) < max_events:
            x_t = self.lookup[cm]
            state = state.add_input(x_t)
            y_t = state.output()
            if APPLY_DROPOUT: y_t = dy.dropout(y_t, self.args.dropout)
            r_t = bias + (R * y_t)
            nm = util.weightedChoice(r_t.vec_value(), range(self.vocab.size), apply_softmax=True)
            events.append(nm)
            if "get_err" in mel_args:
                err = dy.pickneglogsoftmax(r_t, int(nm))
                errs.append(err)
            if nm == self.vocab[self.vocab.END_EVENT].i: break
            cm = nm

        if "get_err" in mel_args:
            err = dy.esum(errs)
            return err
        else:
            return events

