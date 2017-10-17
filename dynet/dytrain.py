
import torch
import torch.nn as nn
from torch.autograd import Variable
import dynet as dy

import numpy as np
import argparse, pylab, io, random, math, time, sys, os
from os import walk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rnnlm, dyrnnlm
import util


def plot_nll(train_losses, dev_losses):
    x = np.arange(len(train_losses))
    train, = pylab.plot(x, train_losses, label="train")
    dev, = pylab.plot(x, dev_losses, label="dev")
    pylab.xlabel("Epoch")
    pylab.ylabel("NLL")
    pylab.legend(handles=[train,dev])
    pylab.show()

def get_loss_value_and_backprop(lm, batch, framework, trainer):
    if framework == 'dynet':
        batch_losses = lm.BuildLMGraph_batch(batch)
        batch_loss = dy.sum_batches(batch_losses)
        batch_loss.backward()
        trainer.update()
        return batch_loss.value()
    else:
        pass

parser = argparse.ArgumentParser()

parser.add_argument("--dynet-mem", help="set size of dynet memory allocation, in MB")
parser.add_argument("--dynet-gpu", help="use GPU acceleration")

# file locations
parser.add_argument("--corpus", default="../music_data/Nottingham/", help="location of entire corpus")
parser.add_argument("--train", default="../music_data/Nottingham/train/", help="location of training data")
parser.add_argument("--valid", default="../music_data/Nottingham/valid/", help="location of validation data")
parser.add_argument("--test", default="../music_data/Nottingham/test/", help="location of test data")

# rnn params
parser.add_argument("--rnn", default="lstm", choices={"lstm","rnn","gru"}, help="choose type of RNN")
parser.add_argument("--layers", default=1, type=int, help="choose number of layers for RNN")
parser.add_argument("--input_dim", default=25, type=int, help="choose token embedding dimension")
parser.add_argument("--hidden_dim", default=100, type=int, help="choose size of hidden state of RNN")
parser.add_argument("--dropout", default=.1, type=float, help="set dropout probability")

# experiment params
parser.add_argument("--trainer", default="sgd", choices={"sgd", "adam", "adagrad"}, help="choose training algorithm")
parser.add_argument("--epochs", default=10, type=int, help="maximum number of epochs to run experiment")
parser.add_argument("--learning_rate", default=0.2, help="set learning rate of trainer")
parser.add_argument("--batch_size", default=16, type=int, help="size of minibatches")

parser.add_argument("--arch", default="baseline", help="choose what RNNLM architecture you want to use")
parser.add_argument("--framework", default="dynet", help="choose what framework to use")

args = parser.parse_args()
batch_size = args.batch_size


###############################################################################
# Load data
###############################################################################

sv = util.SimpleVocab.load_from_corpus(args.corpus, "../tmp/nott_sv.p")

train_data = [util.mid2tuples(f) for f in util.getmidfiles(args.train)]
val_data = [util.mid2tuples(f) for f in util.getmidfiles(args.valid)]
# test_data = [util.mid2tuples(f) for f in util.getmidfiles(args.test)]

train_data = [m for m in train_data if len(m) > 0]
val_data = [m for m in val_data if len(m) > 0]
# test_data = [m for m in test_data if len(m) > 0]

train_data = np.array(train_data)
val_data = np.array(val_data)
# test_data = np.array(test_data)

train_order = range(len(train_data))
val_order = range(len(val_data))
print "Num train melodies:", (len(train_order))
print "Num val melodies:", (len(val_order))


###############################################################################
# Build the model
###############################################################################

if args.framework == 'dynet':
    model = dy.Model()
    if args.trainer == "sgd":
        trainer = dy.SimpleSGDTrainer(model, learning_rate=1.0)
    elif args.trainer == "adam":
        trainer = dy.AdamTrainer(model, learning_Rate=0.001)
    elif args.trainer == "adagrad":
        trainer = dy.AdagradTrainer(model, learning_rate=0.01)
else:
    model = torchrnnlm.RNNModel(args, sv) 

lm = rnnlm.get_model(args.arch + '_' + args.framework)(model, sv, args)


###############################################################################
# Training code
###############################################################################

start = t = time.time()
cum_event_count = cum_loss = 0
train_losses = []
val_losses = []
for epoch in range(args.epochs):

    #### run and log training
    random.shuffle(train_order)
    for i in range(int(len(train_data)/batch_size)):
        # TODO you can step here with range!
        start_idx = batch_size*i
        end_idx = min(len(train_data)-1, batch_size*i + batch_size)
        batch = train_data[train_order[start_idx:end_idx]]
        cum_loss += get_loss_value_and_backprop(lm, batch, 
                args.framework, trainer)
        cum_event_count += sum([len(mel)-1 for mel in batch])

    #### validation 
    random.shuffle(val_order)
    val_event_count = val_loss = 0
    for i in range(int(len(val_data)/batch_size)):
        start_idx = batch_size*i
        end_idx = min(len(val_data)-1, batch_size*i + batch_size)
        v_batch = val_data[val_order[start_idx:end_idx]]
        v_batch_losses = lm.BuildLMGraph_batch(v_batch)
        v_batch_loss = dy.sum_batches(v_batch_losses)
        val_loss += v_batch_loss.value()
        val_event_count += sum([len(mel)-1 for mel in v_batch])

    #### Print stats
    print
    print "Epoch", epoch+1
    print "L:", cum_loss / cum_event_count, "|", val_loss / val_event_count 
    print "P:", math.exp(cum_loss / cum_event_count), "|", math.exp(val_loss / val_event_count)
    print "Time:", time.time() - t, "| Total:", time.time() - start
    print "WPS:", cum_event_count / (time.time() - start)
    t = time.time()
    train_losses.append(cum_loss / cum_event_count )
    val_losses.append(val_loss / val_event_count)

sv.list2mid(lm.sample(), "../../generated/test.mid")
plot_nll(train_losses, val_losses)
lm.save("../models/1")

######## Get test stats (TODO)


