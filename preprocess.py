import argparse
import dill
import nltk
import string

###############################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Runs text preprocessing')
    parser.add_argument('--readfile', required=True, dest='read_path')
    parser.add_argument('--savefile', required=True, dest='save_path')
    parser.add_argument('--statfile', required=True, dest='stat_path')
    args = parser.parse_args()

    punctuation = string.punctuation
    punctuation = punctuation.replace('@','')
    punctuation = punctuation.replace('#','')
    punctuation = punctuation.replace('_','')
    # table = str.maketrans('', '', punctuation) # For python3, comment this line for python2

    # Read the file
    with open(args.read_path, 'r') as f:
        data = f.readlines()
    
    # Preprocess
    for i, line in enumerate(data):
        tweet = line.split('\n')[0]
        tweet = tweet.split()
        # tweet = [word.translate(table) for word in tweet]
        tweet = [word.translate(None, punctuation) for word in tweet] # Only for python2, comment previous line if using python2
        tweet = [word.lower() for word in tweet]
        data[i] = tweet

    # Write required statistics about the data to file
    total_tweets = []
    for tweet in data:
        total_tweets.extend(tweet)
    freq_dist = nltk.FreqDist(total_tweets)
    most_common = freq_dist.most_common(50)
    with open(args.stat_path, 'w') as f:
        f.write("Trump's vocabulary size: " + str(len(freq_dist)) + '\n')
        f.write('\n')
        f.write(20*'*' + '\n')
        f.write('\n')
        f.write('Top 50 most common words:\n')
        f.write('\n')
        for item in most_common:
            f.write(item[0] + ': ' + str(item[1]) + '\n')

    # Save preprocessed data as dill
    dill.dump(data, open(args.save_path, 'wb'))