
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
        num_filters = [64, 128, 256]
        fcneurons = [512, 256, 1]
        self.rewards = tf.placeholder(tf.float32, [None, 1])
        self.states = tf.placeholder(tf.float32, [None, 8, 8, 12 * N + 9])

        conv1 = tf.layers.conv2d(
                    inputs=self.states,
                    filters=num_filters[0],
                    kernel_size=[4, 4],
                    padding='same',
                    activation=tf.nn.relu)
        
        b_conv1 = tf.layers.batch_normalization(conv1)

        conv2 = tf.layers.conv2d(
                    inputs=b_conv1,
                    filters=num_filters[1],
                    kernel_size=[4, 4],
                    padding='same',
                    activation=tf.nn.relu)

        b_conv2 = tf.layers.batch_normalization(conv2)

        conv3 = tf.layers.conv2d(
                    inputs=b_conv2,
                    filters=num_filters[2],
                    kernel_size=[4, 4],
                    padding='same',
                    activation=tf.nn.relu)

        b_conv3 = tf.layers.batch_normalization(conv3)


        conv_flatten = tf.reshape(b_conv3, [-1, 8 * 8 * features[-1]])

        fc1 = tf.layers.dense(inputs=conv_flatten, units=fcneurons[0])
        fc2 = tf.layers.dense(inputs=fc1, units=fcneurons[1])
        self._estimate = tf.layers.dense(inputs=fc2, units=fcneurons[2])

        self._loss = tf.squared_difference(self._estimate, self.rewards)
        self._av_loss = tf.reduce_mean(self._loss)
        self.print_loss = tf.reduce_mean(tf.squared_difference(self._estimate, self.rewards))

        self._optimizer = tf.train.AdagradOptimizer(1E-4).minimize(self._av_loss) 
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
    
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
        if os.path.exists('{}.meta'.format(path)):
            self.saver = tf.train.import_meta_graph('{}.meta'.format(path))
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))

def min_select(boards, model, history):
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

def max_select(boards, model, history):
    if len(boards) == 0:
        boards.append(chess.Board())
    
    curr_board = boards[-1]
    possible = list(curr_board.legal_moves)
    
    max_val = -1 *float('inf')
    for move in possible:
        t_boards = copy.deepcopy(boards)
        t_curr = copy.copy(t_boards[-1])
        t_curr.push(move)
        t_boards.append(t_curr)

        val = model.evaluate(get_input(t_boards, history))
        if val > max_val:
            max_val = val
            best_move = move

    return best_move

def select(boards, model, history):
    return max_select(boards, model, history)

def chess_worker(connection, args):
    while True:
        board = chess.Board()
        player = 1

        boards = []
        while not board.is_game_over(claim_draw=True):

            if random.random() > args.gamma:
                connection.send({'type': 'board', 'boards': boards[-8:]})
                move = connection.recv()
            else:
                move = random.choice(list(board.legal_moves))

            player = (player % 2) + 1

            board.push(move)
            boards.append(copy.copy(board))

        res = board.result()

        if res == '1-0':
            rewards = [ -1 if (i % 2) == 0 else 1 for i in range(len(boards)) ]
        elif res == '0-1':
            rewards = [ 1 if (i % 2) == 0 else -1 for i in range(len(boards)) ]
        else:
            rewards = [ 0 for i in range(len(boards)) ]
        
        rewards = np.expand_dims(np.array(rewards).T, 1)
        in_boards, in_rewards = build_input(boards, rewards, args.history)
        connection.send({'type': 'data', 'inp': in_boards, 'rewards': in_rewards})    
        connection.send({'type': 'end'})

