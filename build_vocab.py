import argparse
import dill
import nltk
import numpy as np

###############################################################################

def FreqDist(corpus):
    tweets = []
    for tweet in corpus:
        tweets.extend(tweet)
    freq_dist = nltk.FreqDist(tweets)
    return freq_dist

###############################################################################

def get_vocab(freq_dist, min_occur):
    vocabulary = [word for word in freq_dist if freq_dist[word]>=min_occur]
    return np.array(vocabulary)

###############################################################################

def build_vocab(corpus, min_occur, save_path):

    # Read the corpus
    corpus = dill.load(open(args.corpus,'rb'))

    # Get the words in the corpus with their number of occurrences
    freq_dist = FreqDist(corpus)

    # Build the vocabulary based on min_occur
    # vocabulary is a numpy array of strings
    vocabulary = get_vocab(freq_dist, min_occur)

    # Dill the vocabulary
    dill.dump(vocabulary, open(save_path,'wb'))

###############################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Builds the necessary vocabulary')
    parser.add_argument('--corpus', required=True, dest='corpus')
    parser.add_argument('--min_occur', type=int, required=True, dest='min_occur')
    parser.add_argument('--savepath', required=True, dest='save_path')
    args = parser.parse_args()

    build_vocab(args.corpus, args.min_occur, args.save_path)