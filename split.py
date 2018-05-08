import sys, os
import argparse
import shutil
import glob
import ast
import random

parser = argparse.ArgumentParser(description='Split into train/dev/test')
parser.add_argument('--data', type=str, default='music_data/CMaj_Nottingham/',
                    help='location of the data corpus to sample from')
args = parser.parse_args()

train = args.data + 'train/'
#os.mkdir(train)
valid = args.data + 'valid/'
#os.mkdir(valid)
test = args.data + 'test/'
#os.mkdir(test)


for f in glob.glob(args.data + "*.mid"):
    print f
    if random.random() < 0.8:
        shutil.move(f, train + os.path.basename(f))
    elif random.random() < 0.9:
        shutil.move(f, valid + os.path.basename(f))
    else:
        shutil.move(f, test + os.path.basename(f))


