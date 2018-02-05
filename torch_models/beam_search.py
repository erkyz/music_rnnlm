"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code modified from Sandeep Subramanian @ MILA
# https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/beam_search.py

import torch
import numpy as np


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.vocab = vocab
        self.pad = vocab.special_events['padding'].i
        self.bos = vocab.special_events['start'].i
        self.eos = vocab.special_events['end'].i
        self.tt = torch.cuda if cuda else torch

        self.channels = []
        for c in range(vocab.num_channels): 
            self.channels.append({})

            # The score for each translation on the beam.
            self.channels[c]["scores"] = self.tt.FloatTensor(size).zero_()

            # The backpointers at each time-step.
            self.channels[c]["prevKs"] = []

            # The outputs at each time-step.
            self.channels[c]["nextYs"] = [self.tt.LongTensor(size).fill_(self.pad)]
            self.channels[c]["nextYs"][0][0] = self.bos

            # The attentions (matrix) for each time.
            self.channels[c]["attn"] = []

    # Get the outputs for the current timestep.
    def get_current_state(self, c):
        """Get state of beam."""
        return self.channels[c]["nextYs"][-1] 

    # Get the backpointers for the current timestep.
    def get_current_origin(self, c):
        """Get the backpointer to the beam at this step."""
        return self.channels[c]["prevKs"][-1]

    #  Given prob over words for every last beam `wordLks` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLks`- list. probs of advancing from the last step (K x words), for each channel
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, wordLks):
        """Advance the beam."""
        for c in range(len(wordLks)):
            num_words = self.vocab.sizes[c]

            # Sum the previous scores.
            if len(self.channels[c]["prevKs"]) > 0:
                beam_lk = wordLks[c] + self.channels[c]["scores"].unsqueeze(1).expand_as(wordLks[c])
            else:
                beam_lk = wordLks[c][0]

            flat_beam_lk = beam_lk.view(-1)

            bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
            self.channels[c]["scores"] = bestScores

            # bestScoresId is flattened beam x word array, so calculate which
            # word and beam each score _came from_
            prev_k = bestScoresId / num_words
            self.channels[c]["prevKs"].append(prev_k)
            self.channels[c]["nextYs"].append(bestScoresId - prev_k * num_words)

            # End condition is when top-of-beam is EOS.
            if self.channels[c]["nextYs"][-1][0].data[0] == self.eos:
                self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        # TODO need to sum over all the channels.
        return torch.sort(self.channels[0]["scores"], 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    def get_hyp(self, k):
        """Get hypotheses."""
        hyps = []
        for c in range(len(self.channels)):
            hyp = []
            for j in range(len(self.channels[c]["prevKs"]) - 1, -1, -1):
                hyp.append(self.channels[c]["nextYs"][j + 1][k].data[0])
                k = self.channels[c]["prevKs"][j][k]
            hyps.append(hyp[::-1])

        return hyps

