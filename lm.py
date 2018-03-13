import numpy as np
import os
import tensorflow as tf

class LanguageModel(object):

    def __init__(self, lr=1e-3, hsizes, num_steps, vocab_len, batch_size):
        self.lr = lr
        self.hidden_sizes = hsizes
        self.vocab_len = vocab_len
        self.num_steps = num_steps
        self.batch_size
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
        self.input = tf.placeholder(shape=(self.batch_size, self.num_steps, None), dtype=tf.float32)

    def length(sequence):
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
    
        self.output, _ = tf.nn.dynamic_rnn(cells, seq, sequenece_length = length(self.input), self.in_state)

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

        # seq = tf.one_hot(self.input, self.vocab_len)
        # forward_prop(seq)
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :-1], 
        #                                                 labels=seq[:, 1:])
        # self.loss = tf.reduce_sum(loss)

    def add_train_op(self):
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def create_summary(self):
        self.summary = tf.summary.scalar('loss', self.loss)

    def train_batch(self, sess, inputs):
        feed_dict = self.create_feed_dict(inputs=inputs)
        loss, _, summary = sess.run([self.loss, self.train, self.summary], feed_dict=feed_dict)
        return loss, summary

    def run_epoch(self, sess, saver, writer, train_data, train_labels):
        num_minibatches = int(len(train_data) / self.batch_size)
        minibatch_train = np.array_split(train_data, num_minibatches)
        minibatch_labels = np.array_split(train_labels, num_minibatches)
        epoch_loss = 0
        for i in range(len(minibatch_train)):
            m_inputs, m_labels = [],[]
            for j in minibatch_train[i]:
                m_inputs.append(j)
            for k in minibatch_labels[i]:
                m_labels.append(k)
            m_inputs = np.array(m_inputs)
            m_labels = np.array(m_labels)
            # Train on the minibatch and add to the summary
            loss, summary = self.train_batch(sess=sess, inputs=m_inputs)
            epoch_loss += loss
            writer.add_summary(summary, global_step=self.minibatch_count)
            self.minibatch_count += 1
        epoch_loss /= float(len(minibatch_train))
        return epoch_loss

    def fit(self, sess, train_data, train_labels, num_epochs=50, folder='./', graph_folder='./'):
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

    # def predict(self, sess, test_data, spa_dict):
    #     inputs = []
    #     for word in test_data:
    #         inputs.append(spa_dict[word])
    #     inputs = np.array(inputs)
    #     feed_dict = self.create_feed_dict(inputs=inputs)
    #     return sess.run(self.output, feed_dict=feed_dict)