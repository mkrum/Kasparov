
import tensorflow as tf
import numpy as np

class Zero(object):

    sess = tf.Session()

    rewards = tf.placeholder(tf.float32, [None, 1])
    states = tf.placeholder(tf.float32, [None, 3, 3, 7])
    probs = tf.placeholder(tf.float32, [None, 3, 3, 1])

    def __init__(self):
        features = [64, 64, 32]
        fcneurons = [64, 32, 1]

        conv1 = tf.layers.conv2d(
                    inputs=self.states,
                    filters=features[0],
                    kernel_size=[3, 3],
                    padding='same',
                    activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=features[1],
                    kernel_size=[3, 3],
                    padding='same',
                    activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=features[2],
                    kernel_size=[3, 3],
                    padding='same',
                    activation=tf.nn.relu)
        
        self._probs = tf.layers.conv2d(
                    inputs=conv3,
                    filters=1,
                    kernel_size=[3, 3],
                    padding='same',
                    activation=tf.nn.sigmoid)

        conv_flatten = tf.reshape(conv3, [-1, 3 * 3 * features[-1]])
        fc1 = tf.layers.dense(inputs=conv_flatten, units=fcneurons[0], activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=fcneurons[1], activation=tf.nn.relu)
        self._value = tf.layers.dense(inputs=fc2, units=fcneurons[2])

        flat_probs = tf.transpose(tf.reshape(self._probs, [-1, 3 * 3]))
        flat_actual = tf.transpose(tf.reshape(self.probs, [-1, 3 * 3]))

        self._loss = tf.squared_difference(self._value, self.rewards) \
                + tf.matmul(tf.transpose(flat_probs), tf.log(flat_actual))

        self._optimizer = tf.train.AdamOptimizer(.2).minimize(self._loss) 
        self.sess.run(tf.global_variables_initializer())
    
    def evaluate(self, game, player):
        probs, values = self.sess.run([self._probs, self._value], feed_dict={self.states: 
                                                                             game.get_input(player)})
        #probs = np.ones((1, 3, 3, 1))
        return probs, values

    def train(self, boards, probs, rewards):
        self.sess.run(self._optimizer, feed_dict={self.states: boards, 
                                                  self.rewards: rewards, self.probs: probs})
