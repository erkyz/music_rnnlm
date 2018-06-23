import os, sys
import glob
import music21
import argparse

# USE PYTHON3 FOR THIS BECAUSE MUSIC21 FOR PYTHON 2 IS OUTDATED.
if sys.version_info[0] != 3:
    print("Use Python3!")
    exit()

# http://nickkellyresearch.com/python-script-transpose-midi-files-c-minor/
# Converting everything into the key of C major or A minor

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, default='music_data',
                    help='location of the music data corpuses')
parser.add_argument('--corpus', type=str, default='Nottingham',
                    help='location of the specific corpus')
parser.add_argument('--dir', type=str, default='train',
                    help='name of specific dir')
args = parser.parse_args()


# major conversions
majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("G-", 6),("G", 5)])
minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("G-", 3),("G", 2),("G#", 1)])

os.chdir('../' + args.base + '/' + args.corpus + '/' + args.dir)
corpus_dir = '../../' + 'CMaj_' + args.corpus
if not os.path.exists(corpus_dir):
    os.mkdir(corpus_dir)
this_dir = '../../' + 'CMaj_' + args.corpus + '/' + args.dir
if not os.path.exists(this_dir):
    os.mkdir(this_dir)
i = 0
for file in glob.glob("*.mid"):
    print (file)
    score = music21.converter.parse(file)

    # Only analyze if the key isn't explicitly labeled.
    key = None
    num_keys = 0
    for part in score:
        for e in part:
            if type(e) is music21.key.Key:
                key = e
                num_keys += 1
        break # only do the first Part for Nottingham
    if num_keys > 1:
        i += 1
    if key is None:
        key = score.analyze('key')
    print ("original", key.tonic.name, key.mode)
    if key.tonic.name + key.mode in {'Cmajor', 'Aminor'}:
        print ('Already in C_M/A_m')
        newFileName = "../../" + "CMaj_" + args.corpus + "/" + args.dir + "/" + file
        score.write('midi', newFileName)
        continue
        
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
        
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]
     
    # TODO handle key changes as separate melodies.
    newscore = score.transpose(halfSteps)
    newFileName = "../../" + "CMaj_" + args.corpus + "/" + args.dir + "/" + file
    newscore.write('midi', newFileName)

print ("number pieces with key changes", i)


