
import random, time, util, os, pickle, math

class SaveableModel(object):
    name = "template"
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.add_params()

    def add_params(self):
        pass

    def sample(self, mel_args={}, nchars=100):
        pass

    def save(self, path):
        if not os.path.exists(path): os.makedirs(path)
        self.model.save(path + "/params")
        with open(path+"/args", "w") as f: pickle.dump(self.args, f)

    @staticmethod
    def load(model, path, load_model_params=True):
        if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
        with open(path+"/args", "r") as f: args = pickle.load(f)
        lm = get_model(args.mode)(model, args)
        if load_model_params: lm.model.load(path+"/params")
        return lm

class SaveableRNNLM(SaveableModel):
    name = "rnnlm_template"
    def __init__(self, model, vocab, args):
        self.model = model
        self.vocab = vocab
        self.args = args
        self.add_params()

    def save(self, path):
        if not os.path.exists(path): os.makedirs(path)
        self.vocab.save(path+"/vocab")
        self.model.save(path + "/params")
        with open(path+"/args", "w") as f: pickle.dump(self.args, f)

    @staticmethod
    def load(model, path, load_model_params=True):
        if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
        vocab = util.SimpleVocab.load(path+"/vocab")
        with open(path+"/args", "r") as f: args = pickle.load(f)
        lm = get_model(args.arch)(model, vocab, args)
        if load_model_params: lm.model.load(path+"/params")
        return lm

def get_model(name):
    for c in util.itersubclasses(SaveableModel):
        if c.name == name: return c
    raise Exception("no language model found with name: " + name)

