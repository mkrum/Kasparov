import os.path
import tensorflow as tf

class DQN(object):

    def __init__(self):

        self.sess = tf.Session()
        features = [64, 32, 32]
        fcneurons = [32, 32, 1]
        self.rewards = tf.placeholder(tf.float32, [None, 1])
        self.states = tf.placeholder(tf.float32, [None, 8, 8, 105])

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

        conv_flatten = tf.reshape(conv3, [-1, 8 * 8 * features[-1]])

        fc1 = tf.layers.dense(inputs=conv_flatten, units=fcneurons[0], activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=fcneurons[1], activation=tf.nn.relu)

        self._estimate = tf.layers.dense(inputs=fc2, units=fcneurons[2], activation=tf.nn.tanh)

        self._loss = tf.squared_difference(self._estimate, self.rewards)
        self.print_loss = tf.reduce_mean(tf.squared_difference(self._estimate, self.rewards))

        self._optimizer = tf.train.AdamOptimizer(1E-4).minimize(self._loss) 
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
    
    def evaluate(self, boards):
        return self.sess.run(self._estimate, feed_dict={self.states: boards})

    def train(self, boards, rewards):
        self.sess.run(self._optimizer,
                             feed_dict={self.states: boards, self.rewards: rewards})

    def save(self, path='./.modelprog'):
        self.saver.save(self.sess, path)

    def load(self, path='./.modelprog'):
        if os.path.exists('{}.meta'.format(path)):
            self.saver = tf.train.import_meta_graph('{}.meta'.format(path))
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))

