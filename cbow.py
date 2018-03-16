import argparse
import dill
import numpy as np
import os
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

###############################################################################

class CBOW(object):
    
    def __init__(self, V, word_vecs, lr):
        self.word_vecs = tf.Variable(initial_value=word_vecs, dtype=tf.float32, name='word_vectors')
        # self.word2idx = word2idx
        # self.idx2word = idx2word
        self.V = V
        # self.D = D
        self.lr = lr
        self.build()
    
    def build(self):
        self.add_placeholders()
        self.forward_prop()
        self.add_loss_op()
        self.add_train_op()
        self.create_summary()
    
    def create_feed_dict(self, inputs, labels=None):
        # inputs and labels are integers corresponding to the indices of words
        feed_dict = {}
        feed_dict[self.inputs] = inputs
        if not labels is None:
            feed_dict[self.labels] = labels
        return feed_dict
    
    def add_placeholders(self):
        self.inputs = tf.placeholder(shape=(None,4), dtype=tf.int32)
        self.labels = tf.placeholder(shape=(None,), dtype=tf.int32)
    
    def forward_prop(self):
        self.word_vecs = tf.nn.embedding_lookup(params=self.word_vecs, ids=self.inputs)
        self.avg_vecs = tf.reduce_mean(self.word_vecs, axis=1)
        self.scores = tf.layers.dense(inputs=self.avg_vecs, units=self.V, kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    def add_loss_op(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.scores))
    
    def add_train_op(self):
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    
    def create_summary(self):
        self.summary = tf.summary.scalar('loss', self.loss)
    
    def train_batch(self, sess, inputs, labels):
        feed_dict = self.create_feed_dict(inputs, labels)
        _, _, summary = sess.run([self.loss, self.train_op, self.summary], feed_dict=feed_dict)
        return summary
    
    def run_epoch(self, sess, minibatch_inputs, minibatch_labels, writer, embed_data_path):
        for i in range(len(minibatch_inputs)):
            summary = self.train_batch(sess, minibatch_inputs[i], minibatch_labels[i])
            writer.add_summary(summary, global_step=self.minibatch_count)
            self.minibatch_count += 1
    
    def fit(self, sess, inputs, minibatch_size=64, num_epochs=100, folder='./', graph_folder='./', embed_data_path='./'):
        saver = tf.train.Saver(max_to_keep=200)
        writer = tf.summary.FileWriter(graph_folder, sess.graph)
        indices = list(range(inputs.shape[1]))
        mid = len(indices) // 2
        other_indices = [idx for idx in indices if idx!=mid]
        labels = inputs[:,mid]
        inputs = inputs[:,other_indices]
        num_minibatches = len(inputs) // minibatch_size
        minibatch_inputs = np.array_split(inputs, num_minibatches)
        minibatch_labels = np.array_split(labels, num_minibatches)
        self.minibatch_count = 0
        for i in range(num_epochs):
            self.run_epoch(sess, minibatch_inputs, minibatch_labels, writer, embed_data_path)
            epoch_folder = os.path.join(folder, 'epoch_'+str(i+1))
            os.mkdir(epoch_folder)
            saver.save(sess, os.path.join(epoch_folder, 'model.ckpt'))
            print('Epoch ' + str(i+1) + ' completed')