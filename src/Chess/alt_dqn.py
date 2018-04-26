''' Rewrite of dqn.py to check for logical errors '''

import os.path
import random

import tensorflow as tf
import numpy as np

import chess

from util import build_input, get_input, get_simple_input


class DQN(object):
    def __init__(self):
        self.sess = tf.train.Session()
        self.saver = tf.train.Saver()

        N = 2
        num_filters = [32, 64, 64]
        kernel_sizes = [[8, 8], [4, 4], [3, 3]]
        strides = [4, 2, 1]
        units = [512, 1]

        self.rewards = tf.placeholder(tf.float32, [-1, 1])
        # self.inputs = tf.placeholder(tf.int16, [-1, 8, 8, 12 * N + 9])
        self.inputs = tf.placeholder(tf.int16, [-1, 8, 8, N])

        conv1 = tf.layers.conv2d(
            inputs=self.inputs,
            filters=num_filters[0],
            kernel_size=kernel_sizes[0],
            strides=strides[0],
            padding='same',
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=num_filters[1],
            kernel_size=kernel_sizes[1],
            strides=strides[1],
            padding='same',
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=num_filters[2],
            kernel_size=kernel_sizes[2],
            strides=strides[2],
            padding='same',
            activation=tf.nn.relu)


        conv_flat = tf.reshape(conv3, [-1, 8 * 8 * num_filters[2]])

        fc = tf.layers.dense(inputs=conv_flat, units=units[0], activation=tf.nn.relu)

        self.output = tf.layers.dense(inputs=fc, units=units[1])

        self.loss = tf.losses.mean_squared_error(self.rewards, self.output)
        self.optimizer = tf.train.RMSPropOptimizer(1E-3).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)


    def evaluate(self, boards):
        return self.sess.run(self.output, feed_dict={self.inputs: boards})


    def train(self, boards, rewards):
        losses = []

        for b, r in zip(boards, rewards):
            feed = {self.rewards: r, self.inputs: b}
            _, loss = self.sess.run((self.optimizer. self.loss), feed_dict=feed)
            losses.append(loss)

        return sum(losses)/len(losses)


    def save(self, path='./.modelprog'):
        self.saver(self.sess, path)


    def load(self, path='./.movelprog'):
        meta_path = '{}/.modelprog.meta'.format(path)
        if os.path.exists(meta_path):
            self.saver = tf.train.import_meta_graph(meta_path)
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        else:
            print('{} does not exist'.format(meta_path))


def min_select(boards, model, history):
    ''' select move with lowest value of all possible moves  '''
    if len(boards) == 0:
        boards.append(chess.Board())

    current = boards[-1]

    best_move = None
    best_result = float('inf')

    for move in current.legal_moves:
        next_board = current.copy().push(move)
        boards.append(next_board)
        inputs = np.expand_dims(get_simple_input(boards, history), 0)
        result = model.evaluate(inputs)

        if result < best_result:
            best_move = move
            best_result = result

        boards.pop()

    return best_move


def max_select(boards, model, history):
    ''' select move with highest value of all possible moves  '''
    if len(boards) == 0:
        boards.append(chess.Board())

    current = boards[-1]

    best_move = None
    best_result = float('-inf')

    for move in current.legal_moves:
        next_board = current.copy().push(move)
        boards.append(next_board)
        inputs = np.expand_dims(get_simple_input(boards, history), 0)
        result = model.evaluate(inputs)

        if current.is_checkmate():
            return move

        if result > best_result:
            best_move = move
            best_result = result

        boards.pop()

    return best_move


def select(boards, model, history):
    ''' calls max select '''
    return max_select(boards, model, history)


def chess_worker(connection, args):
    ''' spawns thread to play game '''
    while True:
        board = chess.Board()

        boards = [board]
        while not board.is_game_over(claim_draw=True):
            if random.random() > args.gamma:
                connection.send({'type': 'board', 'boards': boards[-8:]})
                move = connection.recv()
            else:
                move = random.choice(list(board.legal_moves))

            board.push(move)
            boards.append(board.copy())

        res = board.result()

        if res == '1-0' or res == '0-1':
            rewards = [0 if (i % 2 == 0) else 1 for i in range(len(boards) - 1)]
        else:
            rewards = [0 for _ in boards]

        rewards = np.expand_dims(np.array(rewards[::-1]).T, 1)
        in_boards, in_rewards = build_input(boards, rewards, args.history)

        connection.send({'type': 'data', 'inp': in_boards, 'rewards': in_rewards})
        connection.send({'type': 'end'})
