import argparse
import dill
import numpy as np

###############################################################################

def generate_training_examples(corpus, word2idx, window_len, save_path):
    tweets = dill.load(open(corpus,'rb'))
    word2idx = dill.load(open(word2idx,'rb'))
    training_samples = []
    for tweet in tweets:
        if len(tweet) >= window_len:
            start_idx = 0
            end_idx = len(tweet) - window_len + 1
            for i in range(start_idx, end_idx):
                temp_window = tweet[i:i+window_len]
                temp_indices = [word2idx[word] for word in temp_window]
                training_samples.append(temp_indices)
    training_samples = np.array(training_samples)
    dill.dump(training_samples, open(save_path,'wb'))

###############################################################################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Generates training examples for word vectors')
    parser.add_argument('--corpus', required=True, dest='corpus')
    parser.add_argument('--word2idx', required=True, dest='word2idx')
    parser.add_argument('--window_len', type=int, required=True, dest='window_len')
    parser.add_argument('--savepath', required=True, dest='save_path')
    args = parser.parse_args()
    
    generate_training_examples(args.corpus, args.word2idx, args.window_len, args.save_path)