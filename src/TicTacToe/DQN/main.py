
import sys
sys.path.append('..')

from ttt import TicTacToe
import numpy as np
import random
import tensorflow as tf
import copy
from model import DQN
import matplotlib.pyplot as plt


def decay_reward(reward, size, decay=.9):
    ''' 
    Calculates the decay reward for each timestep
    '''

    rewards = [ reward * (decay ** i)  for i in range(size)]
    rewards.reverse()

    return rewards

def random_choice(board):
    '''
    Random selection algorithm
    '''

    possible = np.where(board == 0)
    choice = np.random.random_integers(low=0, high=(possible[0].size - 1), size=1)

    return (possible[0][choice[0]], possible[1][choice[0]])


def debug_run(model):
    '''
    Shows the board and value for each step in a game
    '''
    game = TicTacToe()
        
    pid = np.random.random_integers(low=1, high=2, size=1)[0]
    winner = None
    while winner is None:

        board = game.get_board(pid)

        val = model.evaluate(board.reshape(1, 3, 3))
        print(board)
        print(val)

        x, y = q_select(board, model)

        game.place(pid, x, y)
        winner = game.check_win()

        pid = (pid % 2) + 1

def play_user(model):
    '''
    Test the model against human skill level
    '''
    game = TicTacToe()
        
    pid = np.random.random_integers(low=1, high=2, size=1)[0]
    winner = None
    while winner is None:

        board = game.get_board(pid)
        val = model.evaluate(board.reshape(1, 3, 3))
        print(board)
        
        if pid == 2:
            x, y = q_select(board, model)
        else:
            x = int(input('x: '))
            y = int(input('y: '))

        game.place(pid, x, y)
        winner = game.check_win()

        pid = (pid % 2) + 1



def q_select(board, model):
    '''
    Simple selection algorithm

    Picks the move on the baord with the highest estimated value 
    '''
    possible = np.where(board == 0)
    
    max_val = -1 * float('inf')
    for x, y in zip(possible[0], possible[1]):
        t_board = copy.copy(board)
        t_board[x, y] = 1.0

        val = model.evaluate(t_board.reshape(1, 3, 3))
        if val > max_val:
            max_val = val
            max_x = x
            max_y = y

    return max_x, max_y


def test_against_random(model):
    '''
    Evaluate the model against random performance
    '''
    
    wins = {0: 0, 1: 0, 2: 0}
    
    for _ in range(1000):
        game = TicTacToe()
        
        pid = np.random.random_integers(low=1, high=2, size=1)[0]
        winner = None
        while winner is None:

            board = game.get_board(pid)

            if pid == 1:
                x, y = random_choice(board)
            else:
                x, y = q_select(board, model)

            game.place(pid, x, y)

            winner = game.check_win()

            pid = (pid % 2) + 1

        wins[winner] += 1

    print('Wins: %d Ties: %d Losses: %d' % (wins[2], wins[0], wins[1]))
    return (wins[2] / (wins[0] + wins[1] + wins[2]))


model = DQN()

games = 0
gamma = 1.0

EPOCH = 100
TEST_FRQ = 50 * EPOCH
win_rate = []

while True:
    games += 1

    #play game
    game = TicTacToe()

    boards = {1: [], 2: []}

    pid = 1
    opp_pid = 2
    winner = None
    while winner is None:
        
        board = game.get_board(pid)

        if random.random() < gamma:
            x, y = random_choice(board)
        else:
            x, y = q_select(board, model)

        game.place(pid, x, y)
        board = game.get_board(pid)
        boards[pid].append(board)
        winner = game.check_win()

        pid = (pid % 2) + 1
        
    board = game.get_board(pid)
    boards[pid].append(board)

    if winner != 0:
        loser = (winner % 2) + 1

        #winner_rewards = decay_reward(1, len(boards[winner]))
        #loser_rewards = decay_reward(-1, len(boards[loser]))
        winner_rewards = [1] * len(boards[winner])
        loser_rewards = [-1] * len(boards[loser])

        rewards = np.array(winner_rewards + loser_rewards)
        boards = np.concatenate([boards[winner], boards[loser]])

    else:
        #tie
        one_rewards = [-.5] * len(boards[1])
        two_rewards = [-.5] * len(boards[2])

        rewards = np.array(one_rewards + two_rewards)
        boards = np.concatenate([boards[1], boards[2]])
    
    rewards = rewards.reshape(rewards.shape[0], -1)
    model.train(boards, rewards)

    if games % EPOCH == 0:
        gamma *= .9
        win_rate.append(test_against_random(model))

        debug_run(model)

    if games % TEST_FRQ == 0:
        print(win_rate)
        plt.plot(win_rate)
        plt.ylabel('Winning Percentage')
        plt.xlabel('Epochs')
        plt.show()
        exit()
    
