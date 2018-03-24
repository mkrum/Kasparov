import sys
sys.path.append('..')

from ttt import TicTacToe
import numpy as np
import random
import tensorflow as tf
import copy
from mcts import evaluate, MCTS
from model import Zero

def random_choice(board):
    '''
    Random selection algorithm
    '''

    possible = np.where(board == 0)
    choice = np.random.random_integers(low=0, high=(possible[0].size - 1), size=1)

    return (possible[0][choice[0]], possible[1][choice[0]])

def play_user(model):
    '''
    Test the model against human skill level
    '''
    game = TicTacToe(3)
        
    pid = np.random.random_integers(low=1, high=2, size=1)[0]
    winner = None

    while winner is None:

        board = game.get_board(pid)
        print(board)
        
        if pid == 2:
            x, y, prob = evaluate(model, game, pid, tau=.1)
            print(prob)
            print(model.evaluate(game.get_input(pid)))
        else:
            x = int(input('x: '))
            y = int(input('y: '))

        game.place(pid, x, y)
        winner = game.check_win()

        pid = (pid % 2) + 1

    print(game.get_input(1))

def test_against_random(model, size=100):
    '''
    Evaluate the model against random performance
    '''
    
    wins = {0: 0, 1: 0, 2: 0}
    
    for _ in range(size):
        game = TicTacToe(3)
        
        pid = np.random.random_integers(low=1, high=2, size=1)[0]
        winner = None
        while winner is None:

            board = game.get_board(pid)

            if pid == 1:
                x, y = random_choice(board)
            else:
                r_board = game.get_board_raw()
                x, y, _ = evaluate(model, game, pid, tau=1)

            game.place(pid, x, y)

            winner = game.check_win()

            pid = (pid % 2) + 1

        wins[winner] += 1

    print('Wins: %d Ties: %d Losses: %d' % (wins[2], wins[0], wins[1]))
    return (wins[2] / (wins[0] + wins[1] + wins[2]))

def build_input(dict_list):
    '''
    concatenate the matrix inputs into a single higher dimensional matrix for the model
    '''
    total = dict_list[1][0]
    for j in range(1, len(probs[1])):
        total = np.concatenate((total, dict_list[1][j]))

    for j in range(len(probs[2])):
        total = np.concatenate((total, dict_list[2][j]))
    
    return total

if __name__ == '__main__': 
    
    model = Zero()
    EPOCH = 300
    ITER = 30

    for it in range(ITER):
        for i in range(EPOCH):
            print('{}: {}/{}'.format(it, i, EPOCH), end='\r')
            game = TicTacToe(3)
            
            pid = np.random.random_integers(low=1, high=2, size=1)[0]
            winner = None

            inputs = {1: [], 2: []}
            probs = {1 : [], 2: []}
            while winner is None:

                board = game.get_board(pid)

                r_board = game.get_board_raw()
                x, y, prob = evaluate(model, game, (pid % 2) + 1)

                inputs[pid].append(copy.copy(game.get_input(pid)))
                probs[pid].append(prob)

                game.place(pid, x, y)
                winner = game.check_win()

                pid = (pid % 2) + 1
            
            if winner != 0:
                one_reward = [1 if winner == 1 else -1] * len(inputs[1])
                two_reward = [1 if winner == 2 else -1] * len(inputs[2])
            else:
                one_reward = [0] * len(inputs[1])
                two_reward = [0] * len(inputs[2])

            total_inputs = build_input(inputs)
            total_prob = build_input(probs)
            total_rewards = np.array(one_reward + two_reward)
            total_rewards = total_rewards.reshape((total_rewards.size, 1))
            model.train(total_inputs, total_prob, total_rewards)


        test_against_random(model)

    test_against_random(model, 1000)
