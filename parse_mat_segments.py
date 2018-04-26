# Parse the matlab-miditoolbox-created segments in a dataset
# Save to new metas.p file

import pickle

d = 'music_data/ashover/test/'

metas = {}
with open(d + 'segments.txt') as segs:
    for l in segs:
        if l[0] != '1':
            f = l[:-1]
        else:
            starts = [int(x)-1 for x in l.split(' ')[:-1]]
            ends = starts[1:] + [-1] # last end shouldn't matter.
            metas[f] = {'segments': list(zip(starts, ends))}

pickle.dump(metas, open(d + 'metas.p', 'wb'))

