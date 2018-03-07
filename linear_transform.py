import numpy as np
import os
import tensorflow as tf

class Transform(object):

    def __init__(lr=1e-3):
        self.lr = lr
        self.build()

    def build(self):
        self.add_placeholders()
        self.forward_prop()
        self.add_loss_op()
        self.add_train_op()
        self.create_summary()

    def create_feed_dict(self, inputs, labels=None):
        feed_dict = {}
        feed_dict[self.inputs] = inputs
        if not labels is None:
            feed_dict[self.labels] = labels
        return feed_dict

    def add_placeholders(self, inputs, labels):
        self.inputs = tf.placeholder(shape=(None,300), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(None,300), dtype=tf.float32)

    def forward_prop(self):
        self.output = tf.layers.dense(inputs=self.inputs, units=300, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def add_loss_op(self):
        self.loss = tf.reduce_mean(tf.square(self.output - self.labels))

    def add_train_op(self):
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def create_summary(self):
        self.summary = tf.summary.scalar('loss', self.loss)

    def predict(self, sess, test_data, spa_dict):
        inputs = []
        for word in test_data:
            inputs.append(spa_dict[word])
        inputs = np.array(inputs)
        feed_dict = self.create_feed_dict(inputs=inputs)
        return sess.run(self.output, feed_dict=feed_dict)

    def train_batch(self, sess, inputs, labels):
        feed_dict = self.create_feed_dict(inputs=inputs, labels=labels)
        _, _, summary = sess.run([self.loss, self.train, self.summary], feed_dict=feed_dict)
        return summary

    def run_epoch(self, sess, saver, writer, train_data, eng_dict, spa_dict, minibatch_size):
        num_minibatches = int(len(train_data) / minibatch_size)
        minibatch_data = np.array_split(train_data, num_minibatches)
        for i in len(minibatch_data):
            m_inputs, m_labels = [], []
            for word_pair in minibatch_data[i]:
                m_inputs.append(spa_dict[word_pair[0]])
                m_labels.append(eng_dict[word_pair[1]])
            m_inputs, m_labels = np.array(m_inputs), np.array(m_labels)
            # Train on the minibatch and add to the summary
            summary = self.train_batch(sess=sess, inputs=m_inputs, labels=m_labels)
            writer.add_summary(self.summary)

    def fit(self, sess, train_data, eng_dict, spa_dict, minibatch_size=64, num_epochs=50, folder='./'):
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graphs', sess.graph)
        for i in range(num_epochs):
            epoch_folder = folder + 'epoch_' + str(i+1) + '/'
            os.makedir(epoch_folder)
            self.run_epoch(sess, saver, writer, train_data, eng_dict, spa_dict, minibatch_size)
            saver.save(sess, epoch_folder+'model.ckpt')