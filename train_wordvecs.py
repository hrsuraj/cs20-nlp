import argparse
import dill
import numpy as np
import os
import tensorflow as tf

from cbow import CBOW

###############################################################################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Trains word vectors with CBOW model')
    parser.add_argument('-m', required=True, choices=['train', 'test'], dest='mode')
    parser.add_argument('--init_vec', required=False, dest='init_vec')
    parser.add_argument('--train_samples', required=False, dest='samples')
    parser.add_argument('--lr', type=float, default=1e-3, dest='lr')
    parser.add_argument('--minibatch_size', type=int, default=64, dest='minibatch_size')
    parser.add_argument('--num_epochs', type=int, default=100, dest='num_epochs')
    parser.add_argument('--models_folder', default='./word_vector_models', dest='folder')
    parser.add_argument('--graph_folder', default='./trump_graph', dest='graphs')
    args = parser.parse_args()
    
    # Read the initial word vectors
    word_vectors = dill.load(open(args.init_vec,'rb'))
    
    cbow = CBOW(len(word_vectors), word_vectors, args.lr)
    init = tf.global_variables_initializer()
    
    # Fit the model
    if args.mode == 'train':
        # Read training samples
        inputs = dill.load(open(args.samples,'rb'))
        with tf.Session() as sess:
            sess.run(init)
            cbow.fit(sess, inputs, minibatch_size=args.minibatch_size, num_epochs=args.num_epochs, folder=args.folder, graph_folder=args.graphs)
    else:
        with tf.Graph().as_default():
            model = cbow
            saver = tf.train.Saver()
            folder = args.models_folder
            with tf.Session() as sess:
                saver.restore(sess, os.path.join(folder, 'model.ckpt'))
                word_vecs = model.init_vec
                dill.dump(word_vecs, open('word_vecs', 'wb'))
