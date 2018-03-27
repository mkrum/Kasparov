from __future__ import print_function

import random
import chess
import copy
import threading
import multiprocessing

import chess.uci
import os
from util import get_input
import numpy as np
from model import DQN

def build_input(boards):
    size = len(boards)
    inputs = np.zeros((size, 8, 8, 105))

    for i in range(1, len(boards) + 1):
        inputs[(i - 1), :, :, :] = get_input(boards[:i])

    return inputs

def q_select(boards, model):
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

        val = model.evaluate(get_input(t_boards))
        if val > max_val:
            max_val = val
            max_move = move

    return max_move

def test_random(model, size=10):
    wins = 0
    draws = 0
    for i in range(size):
        print('{}/{}'.format(i + 1, size), end='\r')
        board = chess.Board()
        boards = []
        player = 1
        while not board.is_game_over(claim_draw=True):

            if player == 1:
                move = q_select(boards, model)
            else:
                move = random.choice(list(board.legal_moves))

            player = (player % 2) + 1

            board.push(move)
            boards.append(copy.copy(board))

        res = board.result()

        if res == '1-0':
            wins += 1
        else:
            draws += 1
    print('Wins: {} Draws: {} Losses: {}'.format(wins, draws, size - (wins + draws)))

def play_game(model, gamma):
    board = chess.Board()
    player = 1

    boards = []
    while not board.is_game_over(claim_draw=True):
        
        if random.random() > gamma:
            move = q_select(boards, model)
        else:
            move = random.choice(list(board.legal_moves))

        player = (player % 2) + 1

        board.push(move)
        boards.append(copy.copy(board))

    res = board.result()

    if res == '1-0':
        rewards = [ 1 if (i % 2) == 0 else -1 for i in range(len(boards)) ]
    elif res == '0-1':
        rewards = [ -1 if (i % 2) == 0 else 1 for i in range(len(boards)) ]
    else:
        rewards = [ -.5 for i in range(len(boards)) ]
    
    inp = build_input(boards)
    rewards = np.expand_dims(np.array(rewards).T, 1)

    return inp, rewards

class GameThread(threading.Thread):

    def __init__(self, model, gamma):
        threading.Thread.__init__(self)
        self.model = model
        self.gamma = gamma
        self.finished = True

    def run(self):
        inp, reward = play_game(self.model, self.gamma)
        self.model.train(inp, reward)
        self.finished = False

def main():
    model = DQN()
    gamma = 1.0
    epoch = 100
    for _ in range(1):
        for i in range(epoch):
            print('{}/{}'.format(i + 1, epoch), end='\r')
            inp, reward = play_game(model, gamma)
            model.train(inp, reward)
        gamma *= .9
        test_random(model, 2)
    test_random(model, 100)


def main_multithreaded():
    model = DQN()
    gamma = .95

    n = 0
    for _ in range(1):
        threads = []

        for i in range(4):
            threads.append(GameThread(model, gamma))
            threads[-1].start()

        while n < 20:
            for i, thread in enumerate(threads):
                if thread.finished and n < 20:
                    print('{}/20'.format(n + 1), end='\r')
                    n += 1
                    thread.join()
                    threads[i] = GameThread(model, gamma)
                    threads[i].start()

        for thread in threads:
            thread.join()

        gamma *= .9
    test_random(model, 10)


if __name__ == '__main__':
    main()
