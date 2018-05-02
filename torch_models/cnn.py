import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class CNN(nn.Module):
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
        self.fc = nn.Linear(args.batch_size*args.batch_size*32, args.nhid)
        
    def forward(self, x):
        # TODO pad to the same length.

        # This is based on the fact that our inputs are arrays of numpy arrays (bad)
        max_size = max([y.shape[0] for y in x])
        x = [np.pad(y, (0, max_size - y.shape[0]), 'constant', constant_values=0) for y in x]
        x = Variable(torch.FloatTensor(x).unsqueeze(1), requires_grad=False)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

