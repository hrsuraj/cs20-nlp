import numoy as np
import dill

tweets = dill.load(open("tweets", "rb"))
w2i = dill.load(open("w2i","rb"))
tweets_idx = [[w2i[i] for i in item] for item in tweets]

dill.dump(tweets_idx, open("tweets_idx","wb"))
