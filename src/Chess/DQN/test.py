from __future__ import print_function

import random
import chess.pgn
import chess
import copy
import threading
import multiprocessing as mp
import argparse
import sys
import time

import chess.uci
import os
from util import get_input
import numpy as np
from model import DQN

def to_pgn(board):
    game = chess.pgn.Game()
    moves = list(board.move_stack)
    node = game.add_variation(moves[0])

    for i in range(1, len(moves)):
        node = node.add_variation(moves[i])

    print(game, file=open("model.pgn", "w"), end="\n\n")


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


def main():
    model = DQN()
    model.load()

    while True:
        board = chess.Board()
        player = 1

        boards = []
        while not board.is_game_over(claim_draw=True):

            if player == 1:
                move = q_select(boards, model)
            else:
                move = random.choice(list(board.legal_moves))

            player = (player % 2) + 1

            board.push(move)
            boards.append(copy.copy(board))
            if len(boards) > 8:
                boards.pop(0)

        res = board.result()


        to_pgn(board)
        exit()
        if res == '1-0':
            to_pgn(board)
            exit()
        elif res == '0-1':
            print('loss')
        else:
            print('loss/draw, trying again')
        


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    main()
