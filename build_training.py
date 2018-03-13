import numpy as np
import dill

def gen_data():
	tweets = dill.load(open("tweets", "rb"))
	w2i = dill.load(open("w2i","rb"))
	word_vector = dill.load(open("word_vecs","rb"))
	tweets_idx = [[w2i[i] for i in item] for item in tweets]

	train_data = []
	train_labels = []
	for sentence in tweets_idx:
		sent_vec = []
		label_vec = []
		
		for idx in sentence:
			sent_vec.append(word_vector[idx])
		for j in range(1,len(sentence)):
			label_vec.append(word_vector[sentence[j]])
		
		for i in range(max_len - len(sent_vec)):
			sent_vec.append(np.zeros(300,))
		for k in range(max_len - len(label_vec)):
			label_vec.append(np.zeros(300,))
		
		train_data.append(sent_vec)
		train_labels.append(label_vec)

	train_data = np.array(train_data)
	train_labels = np.array(train_labels)

	print train_data.shape
	print train_labels.shape

gen_data()

