import numpy as np
import tensorflow as tf
import dill
from tqdm import tqdm

def test_gather():

	a = tf.random_uniform(shape = [5,5,10])
	idx = np.array([[range(5)] for x in range(5)])
	print idx.shape
	idx = idx.reshape((5,5))
	
	a_oh = tf.one_hot(indices = idx, depth = 10, axis = -1)

	sess = tf.Session()
	print sess.run(a_oh)


def gen_data():
	tweets = dill.load(open("tweets", "rb"))
	w2i = dill.load(open("w2i","rb"))
	word_vector = dill.load(open("word_vecs","rb"))
	tweets_idx = [[w2i[i] for i in item] for item in tweets]
	max_len = max([len(item) for item in tweets])
	vocab_len = len(w2i.keys())

	train_data = []
	train_labels = []
	dummy_vocab = np.zeros(vocab_len,)
	dummy_300 = np.zeros(300,)
	for sentence in tqdm(tweets_idx, total = len(tweets_idx)):
		sent_vec = []
		label_vec = []
		
		for idx in sentence:
			sent_vec.append(word_vector[idx])
		for j in range(1,len(sentence)):
			label_vec.append(sentence[j])
		
		for i in range(max_len - len(sent_vec)):
			sent_vec.append(dummy_300)
		for k in range(max_len - len(label_vec)):
			label_vec.append(vocab_len + 1)
		
		train_data.append(sent_vec)
		train_labels.append(label_vec)

	print train_labels

	train_data = np.array(train_data)
	train_labels = np.array(train_labels)

	print train_data.shape
	print train_labels.shape

	np.save(open("lm_train_data.npy","w"), train_data)
	np.save(open("lm_train_labels.npy","w"), train_labels)

# test_gather()
gen_data()


