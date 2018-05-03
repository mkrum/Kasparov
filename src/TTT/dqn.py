import os.path
import tensorflow as tf
import numpy as np
import random
import copy

from ttt import TicTacToe

from util import build_input, get_simple_input

class DQN(object):

    def __init__(self):
        N = 2
        self.sess = tf.Session()
        features = [64, 128, 256]
        fcneurons = [512, 256, 1]
        self.rewards = tf.placeholder(tf.float32, [None, 1])
        self.states = tf.placeholder(tf.float32, [None, 3, 3, 2])

        conv1 = tf.layers.conv2d(
                    inputs=self.states,
                    filters=features[0],
                    kernel_size=[4, 4],
                    padding='same',
                    activation=tf.nn.elu)
        
        b_conv1 = tf.layers.batch_normalization(conv1)

        conv2 = tf.layers.conv2d(
                    inputs=b_conv1,
                    filters=features[1],
                    kernel_size=[4, 4],
                    padding='same',
                    activation=tf.nn.elu)

        b_conv2 = tf.layers.batch_normalization(conv2)

        conv3 = tf.layers.conv2d(
                    inputs=b_conv2,
                    filters=features[2],
                    kernel_size=[4, 4],
                    padding='same',
                    activation=tf.nn.elu)

        b_conv3 = tf.layers.batch_normalization(conv3)

        conv_flatten = tf.reshape(b_conv3, [-1, 3 * 3 * features[-1]])

        fc1 = tf.layers.dense(inputs=conv_flatten, units=fcneurons[0], activation=tf.nn.elu)
        fc2 = tf.layers.dense(inputs=fc1, units=fcneurons[1], activation=tf.nn.elu)
        self._estimate = tf.layers.dense(inputs=fc2, units=fcneurons[2], activation=tf.nn.sigmoid)

        self._loss = tf.losses.mean_squared_error(self._estimate, self.rewards)
        self._av_loss = tf.reduce_mean(self._loss)
        self.print_loss = tf.reduce_mean(tf.squared_difference(self._estimate, self.rewards))

        self._optimizer = tf.train.AdagradOptimizer(2E-4).minimize(self._av_loss) 
        self.sess.run(tf.global_variables_initializer())

    def evaluate(self, boards):
        return self.sess.run(self._estimate, feed_dict={self.states: boards})

    def train(self, boards, rewards):
        losses = []
        for b, r in zip(boards, rewards): 
            _, loss = self.sess.run([self._optimizer, self._av_loss],
                                 feed_dict={self.states: b, self.rewards: r})

            losses.append(loss)

        return sum(losses)/len(losses)

    def save(self, path='./.modelprog'):
        self.saver.save(self.sess, path)

    def load(self, path='./.modelprog'):
        meta_path = path + '/.modelprog.meta'
        if os.path.exists(meta_path):
            self.saver = tf.train.import_meta_graph(meta_path)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
        else:
            print("File does not exist")
            exit()

def max_select(boards, model, history):
    
    curr_board = boards[-1]
    possible = list(curr_board.legal_moves)
    
    max_val = -1 *float('inf')
    for move in possible:
        t_boards = copy.deepcopy(boards)
        t_curr = copy.deepcopy(t_boards[-1])
        t_curr = t_curr.push(move)
        t_boards.append(t_curr)

        val = model.evaluate(np.expand_dims(get_simple_input(t_boards, history), 0))
        if val > max_val:
            max_val = val
            best_move = move

    return best_move

def select(boards, model, history):
    return max_select(boards, model, history)

def chess_worker(connection, args):

    while True:
        
        game = TicTacToe()

        rewards = []

        pid = 1
        boards = [copy.deepcopy(game)]
        i = 0

        while game.check_win() is None:
            
            if random.random() > args.gamma:
                connection.send({'type': 'board', 'boards': boards[-8:]})
                move = connection.recv()
            else:
                move = random.choice(game.legal_moves)

            game = game.push(move)
            boards.append(copy.deepcopy(game))

            pid = (pid % 2 + 1)
            i += 1
    
        res = game.check_win()
        size = len(boards) - 1

        if res != 0:
            rewards = [1 if (i % 2) == 0 else 0 for i in range(size) ]
            rewards = rewards[::-1]
        else:
            rewards = [0] * (size)
            
        rewards = np.expand_dims(np.array(rewards), 1)
        in_boards, in_targets = build_input(boards, rewards, args.history)

        connection.send({'type': 'data', 'inp': in_boards, 'rewards': in_targets})    
        connection.send({'type': 'end'})
