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
    
    # Fit the model

    if args.mode == 'train':
        # Read the initial word vectors
        train_data = np.load(open('lm_train_data.npy','r'))
        train_labels = np.load(open('lm_train_labels.npy','r'))
        
        lm = LanguageModel(args.lr, args.num_steps, args.vocab_len, args.minibatch_size)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            lm.fit(sess, train_data, train_labels, num_epochs=args.num_epochs, folder=args.folder, graph_folder=args.graphs)
    else:
        tweets = dill.load(open("tweets", "rb"))
        w2i = dill.load(open("w2i","rb"))
        i2w = dill.load(open("i2w","rb"))
        word_vector = dill.load(open("word_vecs","rb"))

        start_wd = "@hillaryclinton"
        inputs = np.array([[word_vector[w2i[start_wd]]]])

        model = LanguageModel(args.lr, args.num_steps, args.vocab_len, args.minibatch_size)
        saver = tf.train.Saver()
        folder = args.folder

        op_words = []
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(folder, 'model.ckpt'))
            for i in range(1):
                feed_dict = model.create_feed_dict(inputs=inputs)   
                probs, next_state = sess.run([model.logits, model.in_state], feed_dict = feed_dict)
                print next_state
                op_words.append(i2w[np.argmax(probs.shape)])
                inputs = np.array([[word_vector[w2i[op_words[-1]]]]])

        # print " ".join(op_words)



