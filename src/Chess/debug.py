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
from util import get_input, to_pgn
from mcts import mcts_evaluate
import numpy as np

import heapq

def find_settings(path):
    settings_file = open(path + '/settings.txt', 'r')
    lines = settings_file.read().splitlines()
    model = lines[9].split()[-1]
    #model = lines[8].split()[-1]
    history = int(lines[-1].split()[-1])
    return model, history

def main(args):
    model = DQN()
    model.load(args.path)

    engine = chess.uci.popen_engine(os.environ['SFSH'])
    engine.uci()
        
    board = chess.Board()
    player = 1

    boards = [copy.deepcopy(board)]

    while not board.is_game_over(claim_draw=True):

        print('CURRENT BOARD')
        print(board)

        if player == 1:
            
            engine.position(board)
            stk_move, _ = engine.go()
            
            vals = []
            board_moves = []
            possible_moves = list(board.legal_moves)

            for move in possible_moves:
                t_boards = copy.deepcopy(boards)
                t_curr = t_boards[-1]
                t_curr.push(move)
                t_boards.append(t_curr)

                board_moves.append(t_curr)
                
                vals.append(model.evaluate(get_input(t_boards, args.history)))

                if stk_move == move:
                    stk_val = vals[-1]
                    stk_board = t_curr
            
            top_5 = heapq.nlargest(3, vals)
            bottom_5 = heapq.nsmallest(3, vals)
            
            print('Stockfish')
            print(stk_board)
            print(stk_val)
            
            print('Top 5')
            for v in top_5:
                print(board_moves[vals.index(v)])
                print(v)
                print()

            print()

            print('Bottom 5')
            for v in bottom_5:
                print(board_moves[vals.index(v)])
                print(v)
                print()

            print()

            
            
            if args.max_select:
                move = possible_moves[vals.index(max(vals))]
            if args.min_select:
                move = possible_moves[vals.index(min(vals))]
            
            cont = input('Continue?')
        else:
            move = random.choice(list(board.legal_moves))

        player = (player % 2) + 1

        board.push(move)
        boards.append(copy.deepcopy(board))

        if len(boards) > 8:
            boards.pop(0)

        res = board.result()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify the training setup.')

    parser.add_argument('--path', metavar='p', type=str, 
                    help='Path to the model you want to test')

    parser.add_argument('--max', dest='max_select', action='store_const',
                        const=True, default=False, help='')

    parser.add_argument('--min', dest='min_select', action='store_const',
                        const=True, default=False, help='Repress any saving')

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    model, history = find_settings(args.path)
    args.history = history
    args.model = model

    if args.model == 'td':
        from temporal import *
    elif args.model == 'dqn':
        from dqn import *
    elif args.model == 'app':
        from apprentice import *
    elif args.model == 's_td':
        from sit_temporal import *
    else:
        print('Model not found: {}'.format(args.model))
        exit()

    main(args)
