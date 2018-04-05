
import tensorflow as tf

class DQN(object):


    def __init__(self):
        self.sess = tf.Session()

        self.rewards = tf.placeholder(tf.float32, [None, 1])
        self.states = tf.placeholder(tf.float32, [None, 3, 3])

        features = [128, 64, 64, 64]
        fcneurons = [64, 32, 1]
        conv_states = tf.expand_dims(self.states, -1)
        
        conv1 = tf.layers.conv2d(
                    inputs=conv_states,
                    filters=features[0],
                    kernel_size=[2, 2],
                    padding='same',
                    activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=features[1],
                    kernel_size=[2, 2],
                    padding='same',
                    activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=features[2],
                    kernel_size=[2, 2],
                    padding='same',
                    activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
                    inputs=conv3,
                    filters=features[3],
                    kernel_size=[2, 2],
                    padding='same',
                    activation=tf.nn.relu)

        conv_flatten = tf.reshape(conv4, [-1, 3 * 3 * features[-1]])

        fc1 = tf.layers.dense(inputs=conv_flatten, units=fcneurons[0], activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=fcneurons[1], activation=tf.nn.relu)

        self._estimate = tf.layers.dense(inputs=fc2, units=fcneurons[2])#, activation=tf.nn.tanh)

        self._loss = tf.squared_difference(self.rewards, self._estimate)

        self._optimizer = tf.train.AdamOptimizer(1E-3).minimize(self._loss) 
        self.sess.run(tf.global_variables_initializer())
    
    def evaluate(self, boards):
        return self.sess.run(self._estimate, feed_dict={self.states: boards})

    def train(self, boards, rewards):
        self.sess.run(self._optimizer, feed_dict={self.states: boards, self.rewards: rewards})

