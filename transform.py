import dill
import numpy as np
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import time

from linear_transform import Transform

###############################################################################

def top1top5accuracy(eng_pred, eng_words, eng_dict, common_words_list):
    accuracy1, accuracy5 = 0, 0
    common_words = [eng_dict[word] for word in common_words_list]
    common_words = np.array(common_words)
    for i in range(eng_pred.shape[0]):
        diff = cosine_similarity(eng_pred[i].reshape(1, len(eng_pred[i])), common_words)
        diff_args = np.argsort(diff.flatten())
        eng_word = eng_words[i]
        pred_word = common_words_list[diff_args[0]]
        if eng_word == pred_word:
            accuracy1 += 1
        if eng_word in [common_words_list[diff_args[j]] for j in range(5)]:
            accuracy5 += 1
    return accuracy1/float(len(eng_pred)), accuracy5/float(len(eng_pred))

###############################################################################

def top1top5words(spa_words, eng_dict, eng_pred, common_words_list):
    dict1, dict5 = {}, {}
    common_words = [eng_dict[word] for word in common_words_list]
    common_words = np.array(common_words)
    for i in range(eng_pred.shape[0]):
        diff = cosine_similarity(eng_pred[i].reshape(1, len(eng_pred[i])), common_words)
        diff_args = np.argsort(diff.flatten())
        top1 = common_words_list[diff_args[0]]
        top5 = [common_words_list[diff_args[j]] for j in range(5)]
        dict1[spa_words[i]] = top1
        dict5[spa_words[i]] = top5
    return dict1, dict5

###############################################################################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Runs linear transform on word vectors')
    parser.add_argument('-m', required=True, choices=['train', 'test'], dest='mode')
    parser.add_argument('--lr', default=1e-3, type=float, dest='lr')
    parser.add_argument('--batchsize', default=64, type=int, dest='minibatch_size')
    parser.add_argument('--epochs', default=10, type=int, dest='num_epochs')
    parser.add_argument('--spa_dict', required=True, dest='spa_dict')
    parser.add_argument('--eng_dict', required=True, dest='eng_dict')
    parser.add_argument('--train_data', dest='train_data')
    parser.add_argument('--test_data', dest='test_data')
    parser.add_argument('--folder', default='./', dest='folder')
    parser.add_argument('--graphs', default='../graphs', dest='graph_folder')
    args = parser.parse_args()

    t1 = time.time()
    spa_dict = dill.load(open(args.spa_dict, 'rb'))
    eng_dict = dill.load(open(args.eng_dict, 'rb'))
    t2 = time.time()
    print('Time to load the two dictionaries: ' + str(t2-t1))
    folder = args.folder

    if args.mode == 'train':
        learning_rate = args.lr
        minibatch_size = args.minibatch_size
        num_epochs = args.num_epochs
        train_data = dill.load(open(args.train_data, 'rb'))
        graph_folder = args.graph_folder
        with tf.Graph().as_default():
            model = Transform(learning_rate)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                model.fit(sess, train_data, eng_dict, spa_dict, minibatch_size, num_epochs, folder, graph_folder)
        t3 = time.time()
        print('Time to train the model: ' + str(t3-t2))
    else:
        test_data = dill.load(open(args.test_data, 'rb'))
        with tf.Graph().as_default():
            model = Transform()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, os.path.join(folder, 'model.ckpt'))
                eng_vectors_predict = model.predict(sess, test_data[:,0], spa_dict)
                dill.dump(eng_vectors_predict, open('eng_predict', 'wb'))
        
        with open('../common_words.txt', 'r') as f:
            common_words_list = f.readlines()
        for i,line in enumerate(common_words_list):
            common_words_list[i] = line.strip()
        # Perform evaluation 1
        print(20*'*')
        a1, a5 = top1top5accuracy(eng_vectors_predict, test_data[:,1], eng_dict, common_words_list)
        print('Top 1 accuracy: ' + str(a1))
        print('Top 5 accuracy: ' + str(a5))
        print(20*'*')
        # Perform evaluation 2
        spanish_words = ['regresar', 'cabra', 'parecer', 'otras', 'encantado', 'lengua', 'mike', 'hables', 'poder']
        corr_eng_words, indices = [], []
        for word in spanish_words:
            index = list(test_data[:,0]).index(word)
            indices.append(index)
            corr_eng_words.append(test_data[index, 1])
        eng_pred = eng_vectors_predict[indices]
        dict1, dict5 = top1top5words(spanish_words, eng_dict, eng_pred, common_words_list)
        for word in spanish_words:
            print(word)
            print('Top 1 closest word: ' + dict1[word])
            print('Top 5 closest words: ' + ', '.join(dict5[word]))
            print(20*'*')