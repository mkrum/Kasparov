import os.path
import tensorflow as tf
import chess
import numpy as np
import random
import copy

from util import build_input, get_input

class DQN(object):

    def __init__(self):
        N = 2
        self.sess = tf.Session()
        features = [128, 64, 64]
        fcneurons = [512, 256, 1]
        self.rewards = tf.placeholder(tf.float32, [None, 1])
        self.states = tf.placeholder(tf.float32, [None, 8, 8, 12 * N + 9])

        conv1 = tf.layers.conv2d(
                    inputs=self.states,
                    filters=features[0],
                    kernel_size=[4, 4],
                    padding='same',
                    activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=features[1],
                    kernel_size=[4, 4],
                    padding='same',
                    activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=features[2],
                    kernel_size=[4, 4],
                    padding='same',
                    activation=tf.nn.relu)

        conv_flatten = tf.reshape(conv3, [-1, 8 * 8 * features[-1]])

        fc1 = tf.layers.dense(inputs=conv_flatten, units=fcneurons[0], activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=fcneurons[1], activation=tf.nn.relu)
        self._estimate = tf.layers.dense(inputs=fc2, units=fcneurons[2], activation=tf.nn.tanh)

        self._loss = tf.squared_difference(self._estimate, self.rewards)
        self._av_loss = tf.reduce_mean(self._loss)
        self.print_loss = tf.reduce_mean(tf.squared_difference(self._estimate, self.rewards))

        self._optimizer = tf.train.AdagradOptimizer(1E-5).minimize(self._loss) 
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
    
    def evaluate(self, boards):
        return self.sess.run(self._estimate, feed_dict={self.states: boards})

    def train(self, boards, rewards):
        _, loss = self.sess.run([self._optimizer, self._av_loss],
                             feed_dict={self.states: boards, self.rewards: rewards})

        return float(loss)

    def save(self, path='./.modelprog'):
        self.saver.save(self.sess, path)

    def load(self, path='./.modelprog'):
        if os.path.exists('{}.meta'.format(path)):
            self.saver = tf.train.import_meta_graph('{}.meta'.format(path))
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))

def select(boards, model, history):
    if len(boards) == 0:
        boards.append(chess.Board())
    
    curr_board = boards[-1]
    possible = list(curr_board.legal_moves)
    
    min_val = float('inf')
    for move in possible:
        t_boards = copy.deepcopy(boards)
        t_curr = copy.copy(t_boards[-1])
        t_curr.push(move)
        t_boards.append(t_curr)

        val = model.evaluate(get_input(t_boards, history))
        if val < min_val:
            min_val = val
            best_move = move

    return best_move

def chess_worker(connection, gamma, history):
    lam = 0.7
    while True:
        board = chess.Board()
        player = 1

        rel_boards = {1: [], 2: []}
        vals = {1: [], 2: []}
        rewards = {1: [], 2: []}
        boards = []

        pid = 1
        while not board.is_game_over(claim_draw=True):

            if random.random() > gamma:
                connection.send({'type': 'board', 'boards': boards[-8:]})
                move = connection.recv()
            else:
                move = random.choice(list(board.legal_moves))

            rel_boards[pid].append(board)
            
            connection.send({'type': 'eval', 'boards': boards[-8:]})
            val = connection.recv()

            vals[pid].append(val)
            rewards[pid].append(0)

            boards.append(copy.copy(board))

            board.push(move)

            if len(boards) > 8:
                boards.pop(0)

            pid = (pid % 2 + 1)

        res = board.result()

        if res == '1-0':
            rewards[1][-1] = 1
            rewards[2][-1] = -1
        elif res == '0-1':
            rewards[1][-1] = -1
            rewards[2][-1] = 1

        targets = {1: [], 2: []}

        for i in range(len(rewards[1]) - 1):
            targets[1].append(rewards[1][i] + lam * vals[1][i + 1])
        targets[1].append(rewards[1][-1])

        for i in range(len(rewards[2]) - 1):
            targets[2].append(rewards[2][i] + lam * vals[2][i + 1])
        targets[2].append(rewards[2][-1])

        inp = build_input(rel_boards[1] + rel_boards[2], history)
        targets = np.expand_dims(np.array(targets[1] + targets[2]), 1)

        connection.send({'type': 'data', 'inp': inp, 'rewards': targets})    
        connection.send({'type': 'end'})
