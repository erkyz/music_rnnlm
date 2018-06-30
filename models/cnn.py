import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class CNN(nn.Module):
    '''
    Very rough CNN, in this case designed to take a SSM, use adaptive average
    pooling to convert it into a fixed size, and then output into the dimensionality
    of the hidden state of an RNN, to be used as the initial hidden state.
    '''
    def __init__(self, args):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(4*4*32, args.nhid)
        
    def forward(self, x, args):
        # This assumes our inputs are arrays of numpy arrays (bad)
        max_size = max([y.shape[0] for y in x])
        # Pad a lot of zeros.
        x = [np.pad(y, (0, max_size - y.shape[0]), 'constant', constant_values=0) for y in x]
        if args.cuda:
            x = Variable(torch.cuda.FloatTensor(x).unsqueeze(1), requires_grad=False)
        else:
            x = Variable(torch.FloatTensor(x).unsqueeze(1), requires_grad=False)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # arrange by bsz
        out = self.fc(out)
        return out

