
from ttt import TicTacToe
import numpy as np
import random
import tensorflow as tf
import copy
from mcts import evaluate

def random_choice(board):
    '''Random selection algorithm
    '''

    possible = np.where(board == 0)
    choice = np.random.random_integers(low=0, high=(possible[0].size - 1), size=1)

    return (possible[0][choice[0]], possible[1][choice[0]])

def play_user():
    '''Test the model against human skill level
    '''
    game = TicTacToe()
        
    pid = np.random.random_integers(low=1, high=2, size=1)[0]
    winner = None

    while winner is None:

        board = game.get_board(pid)
        print(board)
        
        if pid == 2:
            x, y = evaluate(board, pid)
        else:
            x = int(input('x: '))
            y = int(input('y: '))

        game.place(pid, x, y)
        winner = game.check_win()

        pid = (pid % 2) + 1



if __name__ == '__main__': 

    wins = {0: 0, 1: 0, 2: 0}
    
    for _ in range(100):
        game = TicTacToe()
        
        pid = np.random.random_integers(low=1, high=2, size=1)[0]
        winner = None
        while winner is None:

            board = game.get_board(pid)

            if pid == 1:
                x, y = random_choice(board)
            else:
                r_board = game.get_board_raw()
                x, y = evaluate(r_board, pid)

            game.place(pid, x, y)

            winner = game.check_win()

            pid = (pid % 2) + 1

        wins[winner] += 1

    print('Wins: %d Ties: %d Losses: %d' % (wins[2], wins[0], wins[1]))
    play_user()





