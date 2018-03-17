
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
    game = TicTacToe(3)
        
    pid = np.random.random_integers(low=1, high=2, size=1)[0]
    winner = None
    while winner is None:

        board = game.get_board(pid)

        val = model.evaluate(game.get_input(pid))
        print(board)
        print(val)

        x, y = q_select(pid, board, model, game)

        game.place(pid, x, y)
        winner = game.check_win()

        pid = (pid % 2) + 1

    print(model.sess.run(model.probs, feed_dict={model.states: game.get_input(pid)})[0])

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
            x, y = q_select(pid, board, model, game)
        else:
            x = int(input('x: '))
            y = int(input('y: '))

        game.place(pid, x, y)
        winner = game.check_win()

        pid = (pid % 2) + 1



def q_select(player_id, board, model, game):
    '''Simple selection algorithm

    Picks the move on the baord with the highest estimated value 
    '''

    possible = np.where(board == 0)
    
    max_val = -1 * float('inf')
    for x, y in zip(possible[0], possible[1]):
        hyp_game = copy.deepcopy(game)
        hyp_game.place(player_id, x, y)
        val = model.evaluate(hyp_game.get_input(player_id))

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
        game = TicTacToe(3)
        
        pid = np.random.random_integers(low=1, high=2, size=1)[0]
        winner = None
        while winner is None:

            board = game.get_board(pid)

            if pid == 1:
                x, y = random_choice(board)
            else:
                x, y = q_select(pid, board, model, game)

            game.place(pid, x, y)

            winner = game.check_win()

            pid = (pid % 2) + 1

        wins[winner] += 1

    print('Wins: %d Ties: %d Losses: %d' % (wins[2], wins[0], wins[1]))
    return (wins[2] / (wins[0] + wins[1] + wins[2]))


model = DQN()

games = 0
gamma = 1.0

EPOCH = 1000
TEST_FRQ = 10 * EPOCH
win_rate = []

while True:
    games += 1
    #play game
    game = TicTacToe(3)

    inputs = {1: [], 2: []}

    pid = 1
    opp_pid = 2
    winner = None
    while winner is None:
        
        board = game.get_board(pid)

        if random.random() < gamma:
            x, y = random_choice(board)
        else:
            x, y = q_select(pid, board, model, game)

        game.place(pid, x, y)
        board = game.get_board(pid)
        inputs[pid].append(game.get_input(pid))
        winner = game.check_win()

        pid = (pid % 2) + 1
        
    board = game.get_board(pid)
    inputs[pid].append(game.get_input(pid))

    if winner != 0:
        loser = (winner % 2) + 1

        winner_rewards = [1] * len(inputs[winner])
        loser_rewards =  [-1] * len(inputs[loser])

        rewards = np.array(winner_rewards + loser_rewards)

        total_inputs = inputs[winner][0]
        for j in range(1, len(inputs[winner])):
            total_inputs = np.concatenate((total_inputs, inputs[winner][j]), axis=0)

        for j in range(len(inputs[loser])):
            total_inputs = np.concatenate((total_inputs, inputs[loser][j]), axis=0)
            
    else:
        #tie
        one_rewards = [-.5] * len(inputs[1])
        two_rewards = [-.5] * len(inputs[2])

        rewards = np.array(one_rewards + two_rewards)

        total_inputs = inputs[1][0]
        for j in range(1, len(inputs[1])):
            total_inputs = np.concatenate((total_inputs, inputs[1][j]))

        for j in range(len(inputs[2])):
            total_inputs = np.concatenate((total_inputs, inputs[2][j]))

    rewards = rewards.reshape(rewards.shape[0], -1)
    model.train(total_inputs, rewards)

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
        play_user(model)
    
