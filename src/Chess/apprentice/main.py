from __future__ import print_function

import random
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


def chess_worker(connection):

    engine = chess.uci.popen_engine(os.environ['SFSH'])
    engine.uci() 

    while True:
        board = chess.Board()
        player = 1

        boards = [copy.copy(board)]
        while not board.is_game_over(claim_draw=True):

            engine.position(board)
            move, _ = engine.go()
            
            for pos_move in list(board.legal_moves):

                t_boards = copy.deepcopy(boards)
                t_curr = copy.copy(t_boards[-1])
                t_curr.push(pos_move)
                t_boards.append(t_curr)

                if pos_move == move:
                    rewards = [1]
                else:
                    rewards = [0]

                rewards = np.expand_dims(np.array(rewards).T, 1)
                connection.send({'type': 'data', 'inp': build_input(t_boards), 'rewards': rewards})    

            board.push(move)
            if len(boards) > 8:
                boards.pop(0)

            boards.append(copy.copy(board))

        connection.send({'type': 'end'})    
        '''
        if res == '1-0':
            rewards = [ 1 if (i % 2) == 0 else -1 for i in range(len(boards)) ]
        elif res == '0-1':
            rewards = [ -1 if (i % 2) == 0 else 1 for i in range(len(boards)) ]
        else:
            rewards = [ 0 for i in range(len(boards)) ]
        
        inp = build_input(boards)
        rewards = np.expand_dims(np.array(rewards).T, 1)
        '''


def main(args):
    model = DQN()

    if args.load is not None:
        print('Loading {}'.format(args.load))
        model.load(args.load)

    for it in range(args.iter):
        f = open(str(it), 'w')
        f.close()

        conns = []
        processes = []

        for p in range(args.threads):
            parent_conn, child_conn = mp.Pipe()
            conns.append(parent_conn)
            processes.append(mp.Process(target=chess_worker, args=(child_conn, )))
            processes[-1].start()
        
        games = 0
        done = []
        while games < args.epoch:
            for conn in conns:

                if conn.poll():
                    msg = conn.recv()

                    if msg['type'] == 'data':
                        model.train(msg['inp'], msg['rewards'])

                    elif msg['type'] == 'end':
                        games += 1
                        if not args.quiet:
                            print('{}/{}'.format(games, args.epoch), end='\r')

        for p in processes:
            p.terminate()
            p.join()

        model.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify the training setup.')

    parser.add_argument('--threads', metavar='t', type=int, default=1,
                    help='Number of threads.')

    parser.add_argument('--iter', metavar='i', type=int, default=1,
                    help='Number of iterations')

    parser.add_argument('--epoch', metavar='e', type=int, default=100,
                    help='Size of each epoch')

    parser.add_argument('--quiet', dest='quiet', action='store_const',
                        const=True, default=False, help='Repress progress output')

    parser.add_argument('--load', metavar='e', type=str, default=None,
                    help='Load a pre existing file')

    parser.add_argument('--gamma', metavar='e', type=float, default=1.0,
                    help='Exploration parameter')

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    main(args)
