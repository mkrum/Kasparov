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

def find_settings(path):
    settings_file = open(path + '/settings.txt', 'r')
    lines = settings_file.read().splitlines()
    model = lines[9].split()[-1]
    #model = lines[8].split()[-1]
    history = int(lines[-1].split()[-1])
    return model, history

def game_worker(connection, args):
    #setup stockfish
    if args.stockfish > -1:
        engine = chess.uci.popen_engine(os.environ['SFSH'])
        engine.uci()
        engine.setoption({'Skill Level': args.stockfish})
        
    while True:
        board = chess.Board()
        player = 1

        boards = [copy.deepcopy(board)]

        while not board.is_game_over(claim_draw=True):

            if player == 1:
                connection.send({'type': 'board', 'boards': boards[-8:]})
                move = connection.recv()
            else:
                if args.stockfish == -1:
                    move = random.choice(list(board.legal_moves))
                else:
                    engine.position(board)
                    move, _ = engine.go()

            player = (player % 2) + 1

            board.push(move)
            boards.append(copy.deepcopy(board))

            if len(boards) > 8:
                boards.pop(0)

        res = board.result()
        connection.send({'type': 'end', 'result' : res, 'pgn' : to_pgn(board)})


def main(args):
    model = DQN()
    model.load(args.path)
    conns = []
    processes = []
    for p in range(args.threads):
        parent_conn, child_conn = mp.Pipe()
        conns.append(parent_conn)
        processes.append(mp.Process(target=game_worker, args=(child_conn, args)))
        processes[-1].start()
    
    games = 0
    losses = []
    while games < args.games:

        for conn in conns:
            if conn.poll():
                msg = conn.recv()

                if msg['type'] == 'board':
                    if args.mcts == -1:
                        conn.send(select(msg['boards'], model, args.history))
                    else:
                        conn.send(mcts_evaluate(model, msg['boards'], args.mcts, args.history))

                elif msg['type'] == 'end':
                    games += 1
                    print(msg['result'])
                    print(msg['pgn'])

    for p in processes:
        p.terminate()
        p.join()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify the training setup.')

    parser.add_argument('--path', metavar='p', type=str, 
                    help='Path to the model you want to test')

    parser.add_argument('--games', metavar='g', type=int, default=10,
                    help='Number of games you want to test over')

    parser.add_argument('--threads', metavar='t', type=int, default=1,
                    help='Number of threads')

    parser.add_argument('--mcts', metavar='m', type=int, default=-1,
                    help='Use MCTS evaluation')

    parser.add_argument('--stockfish', metavar='o', type=int, default=-1,
                    help='Play against specified level of stockfish')

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
