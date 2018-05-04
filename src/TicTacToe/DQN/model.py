
import tensorflow as tf

class DQN(object):

    sess = tf.Session()

    rewards = tf.placeholder(tf.float32, [None, 1])
    states = tf.placeholder(tf.float32, [None, 3, 3])

    def __init__(self):
        features = [64, 64, 32]
        fcneurons = [32, 32, 1]
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

        conv_flatten = tf.reshape(conv3, [-1, 3 * 3 * features[-1]])

        fc1 = tf.layers.dense(inputs=conv_flatten, units=fcneurons[0], activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=fcneurons[1], activation=tf.nn.relu)

        self._estimate = tf.layers.dense(inputs=fc2, units=fcneurons[2], activation=tf.nn.tanh)

        loss = tf.reduce_mean(tf.squared_difference(self._estimate, self.rewards))

        self._optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss) 
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
    
    def evaluate(self, boards):
        return self.sess.run(self._estimate, feed_dict={self.states: boards})

    def train(self, boards, rewards):
        self.sess.run(self._optimizer, feed_dict={self.states: boards, self.rewards: rewards})

    def save(self, path='./.modelprog'):
        self.saver.save(self.sess, path)

    def load(self, path='./.modelprog'):
        meta_path = path + '/.modelprog.meta'
        if os.path.exists(meta_path):
            self.saver = tf.train.import_meta_graph(meta_path)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
        else:
            print('file does not exist')
            exit()
