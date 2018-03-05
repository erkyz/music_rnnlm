import util
pdv = util.PitchDurationVocab()

###########################################################################
# Measure-based similarity
# I'm not using this right now.
###########################################################################

def get_padded_ssm(melody, measure_length, sv):
    ''' |sv| must be a PDV'''
    melody = [sv.i2e[0][i].original for i in melody][1:]
    differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))

    measures = []
    measure_progress = 0
    measure = []
    for i in range(len(differences)):
        if measure_progress == measure_length or i == len(differences)-1:
            measures.append(measure)
            measure = []
            measure_progress = 0
        measure.append(differences[i][0])
        measure_progress += 1

    dist_matrix = np.zeros([len(measures), len(measures)])
    sum_distances = 0
    for i in xrange(len(measures)):
        for j in xrange(len(measures)):
            dist_matrix[i,j] = edit_distance(measures[i], measures[j])
            if i >= j:
               sum_distances += dist_matrix[i,j] 
    max_dist = np.max(dist_matrix)
    ssm = max_dist - dist_matrix 
    return ssm


def get_ssm(f, pdv):
    melody, measure_limit = pdv.mid2orig(f, include_measure_boundaries=False)
    melody = melody[1:-1]

    first_duration = melody[0][1]
    differences = map(diff, zip([('C0', 0)] + melody[:-1], melody))

    measures = []
    measure_progress = 0
    measure = []
    for x in differences:
        if measure_progress >= measure_limit:
            measures.append(measure)
            measure = []
            measure_progress -= measure_limit
        measure.append(x[0])
        measure_progress += x[1]

    dist_matrix = np.zeros([len(measures), len(measures)])
    sum_distances = 0
    for i in xrange(len(measures)):
        for j in xrange(len(measures)):
            dist_matrix[i,j] = float(edit_distance(measures[i], measures[j]))
            if i >= j:
               sum_distances += dist_matrix[i,j] 
    max_dist = np.max(dist_matrix)
    ssm = max_dist - dist_matrix 
    return ssm

    # print measures
    # print sum_distances / ((len(measures)**2)/2)
    # plt.imshow(dist_matrix, cmap='gray', interpolation='nearest')
    # plt.savefig('../similarities/' + args.melody + '.png')

