import glob
import os
import music21
import argparse

# http://nickkellyresearch.com/python-script-transpose-midi-files-c-minor/
# converting everything into the key of C major or A minor

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, default='music_data',
                    help='location of the data corpuses')
parser.add_argument('--corpus', type=str, default='Nottingham',
                    help='location of the data corpuses')
parser.add_argument('--dir', type=str, default='train',
                    help='name of specific dir')
args = parser.parse_args()


# major conversions
majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("G-", 6),("G", 5)])
minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("G-", 3),("G", 2)])

os.chdir(args.base + '/' + args.corpus + '/' + args.dir)
corpus_dir = '../../' + 'CMaj_' + args.corpus
if not os.path.exists(corpus_dir):
    os.mkdir(corpus_dir)
this_dir = '../../' + 'CMaj_' + args.corpus + '/' + args.dir
if not os.path.exists(this_dir):
    os.mkdir(this_dir)
i = 0
for file in glob.glob("*reels_simple_chords_46.mid"):
    score = music21.converter.parse(file)

    # Only analyze if the key isn't explicitly labeled.
    key = None
    num_keys = 0
    for part in score:
        print len(part.getElementsByClass(music21.key.Key))
        print part.getElementsByClass(music21.key.Key)[0]
        for e in part:
            if type(e) is music21.key.Key:
                key = e
                print key
                num_keys += 1
        break # only do the first Part for Nottingham
    if num_keys > 1:
        i += 1
    if key is None:
        key = score.analyze('key')
    print "original", key.tonic.name, key.mode
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
        
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]
    
    # TODO handle key changes as separate melodies.
    newscore = score.transpose(halfSteps)
    newFileName = "../../" + "CMaj_" + args.corpus + "/" + args.dir + "/" + file
    # newscore.write('midi', newFileName)

print "number pieces with key changes", i


