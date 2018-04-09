
import os.path
import tensorflow as tf
import chess
import numpy as np
import random
import copy

from util import build_input, get_input

class DQN(object):

    def __init__(self):

        self.sess = tf.Session()
        features = [64, 64, 32]
        fcneurons = [32, 32, 1]
        self.rewards = tf.placeholder(tf.float32, [None, 1])
        self.states = tf.placeholder(tf.float32, [None, 8, 8, 105])

        conv1 = tf.layers.conv2d(
                    inputs=self.states,
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




def select(boards, model, history):
    '''
    Simple selection algorithm

    Picks the move on the baord with the highest estimated value 
    '''
    if len(boards) == 0:
        boards.append(chess.Board())
    
    curr_board = boards[-1]
    possible = list(curr_board.legal_moves)
    
    max_val = -1 * float('inf')
    for move in possible:
        t_boards = copy.deepcopy(boards)
        t_curr = copy.copy(t_boards[-1])
        t_curr.push(move)
        t_boards.append(t_curr)

        val = model.evaluate(get_input(t_boards, history))
        if val > max_val:
            max_val = val
            max_move = move

    return max_move


def chess_worker(connection, gamma, history):
    while True:
        board = chess.Board()
        player = 1

        boards = []
        while not board.is_game_over(claim_draw=True):

            if random.random() > gamma:

                connection.send({'type': 'board', 'boards': boards[-8:]})
                move = connection.recv()
            else:
                move = random.choice(list(board.legal_moves))

            player = (player % 2) + 1

            board.push(move)
            boards.append(copy.copy(board))
            if len(boards) > 8:
                boards.pop(0)

        res = board.result()

        if res == '1-0':
            rewards = [ 1 if (i % 2) == 0 else -1 for i in range(len(boards)) ]
        elif res == '0-1':
            rewards = [ -1 if (i % 2) == 0 else 1 for i in range(len(boards)) ]
        else:
            rewards = [ 0 for i in range(len(boards)) ]
        
        inp = build_input(boards, history)
        rewards = np.expand_dims(np.array(rewards).T, 1)

        connection.send({'type': 'data', 'inp': inp, 'rewards': rewards})    

