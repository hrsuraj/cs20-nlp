import numpy as np
import dill
from tqdm import tqdm

def gen_data():
	tweets = dill.load(open("tweets", "rb"))
	w2i = dill.load(open("w2i","rb"))
	word_vector = dill.load(open("word_vecs","rb"))
	tweets_idx = [[w2i[i] for i in item] for item in tweets]
	max_len = max([len(item) for item in tweets])

	train_data = []
	train_labels = []
	for sentence in tqdm(tweets_idx, length = len(tweets_idx)):
		sent_vec = []
		label_vec = []
		
		for idx in sentence:
			sent_vec.append(word_vector[idx])
		for j in range(1,len(sentence)):
			zero_vec = list(np.zeros(len(w2i.keys()),))
			zero_vec[sentence[j]] = 1.0
			label_vec.append(zero_vec)
		
		for i in range(max_len - len(sent_vec)):
			sent_vec.append(np.zeros(300,))
		for k in range(max_len - len(label_vec)):
			label_vec.append(np.zeros(len(w2i.keys()),))
		
		train_data.append(sent_vec)
		train_labels.append(label_vec)

	train_data = np.array(train_data)
	train_labels = np.array(train_labels)

	print train_data.shape
	print train_labels.shape

	print train_data[0,0:2,:]

	np.save(open("lm_train_data.npy","w"), train_data)
	np.save(open("lm_train_labels.npy","w"), train_labels)

gen_data()

