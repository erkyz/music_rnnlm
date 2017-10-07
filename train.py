
import dyrnnlm as rnnlm
import argparse
import util
import pylab
import io
import random
import math
import time
import dynet as dy
from os import walk
import numpy as np

def plot_nll(train_losses, dev_losses):
    x = np.arange(len(train_losses))
    train, = pylab.plot(x, train_losses, label="train")
    dev, = pylab.plot(x, dev_losses, label="dev")
    pylab.xlabel("Epoch")
    pylab.ylabel("NLL")
    pylab.legend(handles=[train,dev])
    pylab.show()


# rnn = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
parser = argparse.ArgumentParser()

# file locations
parser.add_argument("--corpus", default="/Users/ericzhu/Documents/music_data/Nottingham/", help="location of entire corpus")
parser.add_argument("--train", default="/Users/ericzhu/Documents/music_data/Nottingham/train/", help="location of training data")
parser.add_argument("--valid", default="/Users/ericzhu/Documents/music_data/Nottingham/valid/", help="location of validation data")
parser.add_argument("--test", default="/Users/ericzhu/Documents/music_data/Nottingham/test/", help="location of test data")

# rnn params
parser.add_argument("--rnn", default="lstm", choices={"lstm","rnn","gru"}, help="choose type of RNN")
parser.add_argument("--layers", default=1, type=int, help="choose number of layers for RNN")
parser.add_argument("--input_dim", default=25, type=int, help="choose token embedding dimension")
parser.add_argument("--hidden_dim", default=100, type=int, help="choose size of hidden state of RNN")
parser.add_argument("--dropout", default=.1, type=float, help="set dropout probability")

# experiment params
parser.add_argument("--epochs", default=10, type=int, help="maximum number of epochs to run experiment")
parser.add_argument("--learning_rate", help="set learning rate of trainer")
parser.add_argument("--batch_size", default=32, type=int, help="size of minibatches")

parser.add_argument("--arch", default="baseline", help="choose what RNNLM architecture you want to use")

args = parser.parse_args()

sv = util.SimpleVocab.load_from_corpus(args.corpus, "nott_sv.p")
batch_size = args.batch_size

# '''
model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)
lm = rnnlm.get_model(args.arch)(model, sv, args)

'''
# sort by length of melodies
np.sort(train_data, key=lambda x:-len(x))
val_data.sort(key=lambda x:-len(x))
test_data.sort(key=lambda x:-len(x))
'''

######## Load data
train_data = [util.mid2tuples(f) for f in util.getmidfiles(args.train)]
val_data = [util.mid2tuples(f) for f in util.getmidfiles(args.valid)]
test_data = [util.mid2tuples(f) for f in util.getmidfiles(args.test)]

train_data = [m for m in train_data if len(m) > 0]
val_data = [m for m in val_data if len(m) > 0]
test_data = [m for m in test_data if len(m) > 0]

train_data = np.array(train_data)
val_data = np.array(val_data)
test_data = np.array(test_data)

# start/end indices of each batch

train_order = range(len(train_data))
val_order = range(len(val_data))
print "Num train melodies:", (len(train_order))
print "Num val melodies:", (len(val_order))

######## Begin training
start = t = time.time()
cum_event_count = cum_loss = 0
train_losses = []
val_losses = []
for epoch in range(args.epochs):

    #### run and log training
    random.shuffle(train_order)
    for i in range(int(len(train_data)/batch_size)):
        start_idx = batch_size*i
        end_idx = min(len(train_data)-1, batch_size*i + batch_size)
        batch = train_data[train_order[start_idx:end_idx]]
        batch_losses = lm.BuildLMGraph_batch(batch)
        batch_loss = dy.sum_batches(batch_losses)
        cum_loss += batch_loss.value()
        cum_event_count += sum([len(mel)-1 for mel in batch])
        batch_loss.backward()
        trainer.update()

    #### validation 
    random.shuffle(val_order)
    val_event_count = val_loss = 0
    for i in range(int(len(val_data)/batch_size)):
        start_idx = batch_size*i
        end_idx = min(len(val_data)-1, batch_size*i + batch_size)
        batch = val_data[val_order[start_idx:end_idx]]
        batch_losses = lm.BuildLMGraph_batch(batch)
        batch_loss = dy.sum_batches(batch_losses)
        val_loss += batch_loss.value()
        val_event_count += sum([len(mel)-1 for mel in batch])

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

sv.list2mid(lm.sample(), "test.mid")
plot_nll(train_losses, val_losses)

# '''

######## Get test stats

'''
for i in range(10):
    o = generate(m)
    io.vec_to_midi(o, "batch" + str(i) + ".mid")
'''
