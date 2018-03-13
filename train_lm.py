import argparse
import dill
import numpy as np
import os
import tensorflow as tf

from lm import LanguageModel

###############################################################################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Trains Langauge Model')
    parser.add_argument('-m', required=True, choices=['train', 'test'], dest='mode')
    parser.add_argument('--num_steps', default = 56, required=False, dest='num_steps')
    parser.add_argument('--vocab_len', type=float, default=19800, dest='vocab_len')
    parser.add_argument('--lr', type=float, default=1e-3, dest='lr')
    parser.add_argument('--minibatch_size', type=int, default=64, dest='minibatch_size')
    parser.add_argument('--num_epochs', type=int, default=30, dest='num_epochs')
    parser.add_argument('--models_folder', default='../lm_models', dest='folder')
    parser.add_argument('--graph_folder', default='../lm_graph', dest='graphs')
    args = parser.parse_args()
    
    # Read the initial word vectors
    train_data = np.load(open('lm_train_data.npy','r'))
    train_labels = np.load(open('lm_train_labels.npy','r'))
    
    lm = LanguageModel(args.lr, args.num_steps, args.vocab_len, args.minibatch_size)
    init = tf.global_variables_initializer()
    
    # Fit the model
    if args.mode == 'train':
        with tf.Session() as sess:
            lm.fit(sess, train_data, train_labels, num_epochs=args.num_epochs, folder=args.folder, graph_folder=args.graphs)
    # else:
    #     model = cbow
    #     saver = tf.train.Saver()
    #     folder = args.folder
    #     with tf.Session() as sess:
    #         saver.restore(sess, os.path.join(folder, 'model.ckpt'))
    #         word_vecs = sess.run(model.init_vec)
    #         dill.dump(word_vecs, open('word_vecs', 'wb'))