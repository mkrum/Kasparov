from __future__ import print_function

import random
import chess
import copy

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
        n = 1
        while not board.is_game_over(claim_draw=True):
            print(n)
            n += 1

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

def main():
    model = DQN()
    gamma = 1.0
    for _ in range(2):
        for i in range(100):
            print('{}/100'.format(i + 1), end='\r')
            inp, reward = play_game(model, gamma)
            model.train(inp, reward)
        gamma *= .9
        test_random(model, 2)

if __name__ == '__main__':
    main()
