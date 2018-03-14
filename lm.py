import numpy as np
import os
import tensorflow as tf

class LanguageModel(object):

    def __init__(self, lr=1e-3, num_steps = 56, vocab_len = 19800, batch_size = 64):
        self.lr = lr
        self.hidden_sizes = [300, 300]
        self.vocab_len = vocab_len
        self.num_steps = num_steps
        self.batch_size = batch_size
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

    def add_placeholders(self):
        self.inputs = tf.placeholder(shape=(self.batch_size, self.num_steps, 300), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(self.batch_size, self.num_steps, self.vocab_len), dtype=tf.float32)

    def length_max(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def forward_prop(self):

        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        zero_states = cells.zero_state(self.batch_size, dtype=tf.float32)
        self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]]) 
                                for state in zero_states])
    
        self.output, _ = tf.nn.dynamic_rnn(cells, self.inputs, self.length_max(self.inputs), self.in_state)

        self.logits = tf.layers.dense(self.output, self.vocab_len, activation=None)


    def add_loss_op(self):
        # Compute cross entropy for each frame.
        cross_entropy = self.labels * tf.log(self.logits)
        cross_entropy = -tf.reduce_sum(cross_entropy, 2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.labels), 2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, 1)
        cross_entropy /= tf.reduce_sum(mask, 1)
        self.loss = tf.reduce_mean(cross_entropy)

    def add_train_op(self):
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def create_summary(self):
        self.summary = tf.summary.scalar('loss', self.loss)

    def train_batch(self, sess, inputs, labels):
        feed_dict = self.create_feed_dict(inputs=inputs, labels = labels)
        loss, _, summary = sess.run([self.loss, self.train, self.summary], feed_dict=feed_dict)
        return loss, summary

    def run_epoch(self, sess, saver, writer, train_data, train_labels):
        num_minibatches = int(train_data.shape[0] / float(self.batch_size))
        minibatch_train = np.array_split(train_data, num_minibatches)
        minibatch_labels = np.array_split(train_labels, num_minibatches)
        print minibatch_train.shape
        print minibatch_labels.shape
        epoch_loss = 0
        for i in range(len(minibatch_train)):
            m_inputs, m_labels = [],[]
            for j in minibatch_train[i]:
                m_inputs.append(j)
            for k in minibatch_labels[i]:
                sent_labs = []
                for idx in k:
                    zero_vec = np.zeros(shape = (self.vocab_len,))
                    if idx < self.vocab_len:
                        zero_vec[idx] = 1.0
                    sent_labs.append(zero_vec)
                m_labels.append(sent_labs)
            m_inputs = np.array(m_inputs)
            m_labels = np.array(m_labels)
            print m_labels.shape
            # Train on the minibatch and add to the summary
            loss, summary = self.train_batch(sess=sess, inputs=m_inputs, labels = m_labels)
            epoch_loss += loss
            writer.add_summary(summary, global_step=self.minibatch_count)
            self.minibatch_count += 1
        epoch_loss /= float(len(minibatch_train))
        return epoch_loss

    def fit(self, sess, train_data, train_labels, num_epochs, folder='./', graph_folder='./'):
        saver = tf.train.Saver(max_to_keep=100)
        writer = tf.summary.FileWriter(graph_folder, sess.graph)
        epoch_loss, self.minibatch_count = [], 0
        for i in range(num_epochs):
            epoch_folder = os.path.join(folder, 'epoch_'+str(i+1))
            os.mkdir(epoch_folder)
            epoch_loss.append(self.run_epoch(sess, saver, writer, train_data, train_labels))
            saver.save(sess, os.path.join(epoch_folder,'model.ckpt'))
        with open(os.path.join(folder,'loss.txt'), 'w') as f:
            for val in epoch_loss:
                f.write(str(val) + '\n')
