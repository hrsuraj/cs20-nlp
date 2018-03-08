import argparse
import dill
import numpy as np

from collections import defaultdict

###############################################################################

def word2idx(vocab):
    mapping = defaultdict(int)
    for idx, word in enumerate(vocab):
        mapping[word] = idx
    return mapping

###############################################################################

def idx2word(vocab):
    mapping = defaultdict(str)
    for idx, word in enumerate(vocab):
        mapping[idx] = word
    return mapping

###############################################################################

def initialize_word_vectors(vocab, dim):
    return np.random.rand(vocab.shape[0], dim)

###############################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gets the mapping from indices to words and vice versa and initializes word vectors')
    parser.add_argument('--vocab', required=True, dest='vocab')
    parser.add_argument('--dimension', type=int, required=True, dest='dim')
    parser.add_argument('--save_w2i', required=True, dest='path_w2i')
    parser.add_argument('--save_i2w', required=True, dest='path_i2w')
    parser.add_argument('--save_vec', required=True, dest='path_vec')
    args = parser.parse_args()

    # Read the vocabulary file to get a numpy array of strings
    vocab = dill.load(open(args.vocab, 'rb'))

    # Get mapping from words to indices
    word2idx_map = word2idx(vocab)

    # Get mapping from indices to words
    idx2word_map = idx2word(vocab)

    # Initialize word vectors to random values
    # Returns numpy array of shape - (V, D)
    # where V is the vocabulary size and D is the required dimension of the word vectors
    word_vectors = initialize_word_vectors(vocab, args.dim)

    # Save the maps and the initialized word vectors onto disk using dill
    dill.dump(word2idx_map, open(args.path_w2i,'wb'))
    dill.dump(idx2word_map, open(args.path_i2w,'wb'))
    dill.dump(word_vectors, open(args.path_vec,'wb'))