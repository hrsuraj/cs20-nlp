import dill
import numpy as np
import os
import tensorflow as tf
import argparse

from linear_transform import Transform

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
    parser.add_argument('--graphs', default='./', dest='graph_folder')
    args = parser.parse_args()

    spa_dict = dill.load(open(args.spa_dict, 'rb'))
    eng_dict = dill.load(open(args.eng_dict, 'rb'))
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
    else:
        test_data = dill.load(open(args.test_data, 'rb'))
        with tf.Graph().as_default():
            model = Transform()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, os.path.join(folder, 'model.ckpt'))
                eng_vectors_predict = model.predict(sess, test_data[:,0], spa_dict)
                dill.dump(eng_vectors_predict, open('eng_predict', 'wb'))