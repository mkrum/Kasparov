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

def play_game(gamma):

    model = DQN()

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
        rewards = [ 0 for i in range(len(boards)) ]
    
    inp = build_input(boards)
    rewards = np.expand_dims(np.array(rewards).T, 1)

    return inp, rewards


def chess_worker(connection, gamma):
    while True:
        board = chess.Board()
        player = 1

        boards = []
        while not board.is_game_over(claim_draw=True):

            if random.random() > gamma:

                connection.send({'type': 'board', 'boards': boards[-8:]})
                move = connection.recv()
            else:
                move = random.choice(list(board.legal_moves))

            player = (player % 2) + 1

            board.push(move)
            boards.append(copy.copy(board))
            if len(boards) > 8:
                boards.pop(0)

        res = board.result()

        if res == '1-0':
            rewards = [ 1 if (i % 2) == 0 else -1 for i in range(len(boards)) ]
        elif res == '0-1':
            rewards = [ -1 if (i % 2) == 0 else 1 for i in range(len(boards)) ]
        else:
            rewards = [ 0 for i in range(len(boards)) ]
        
        inp = build_input(boards)
        rewards = np.expand_dims(np.array(rewards).T, 1)

        connection.send({'type': 'data', 'inp': inp, 'rewards': rewards})    


def main(args):
    model = DQN()
    if args.load is not None:
        print('Loading {}'.format(args.load))
        model.load(args.load)

    gamma = args.gamma
    for it in range(args.iter):
        f = open(str(it), 'w')
        f.close()

        conns = []
        processes = []

        for p in range(args.threads):
            parent_conn, child_conn = mp.Pipe()
            conns.append(parent_conn)
            processes.append(mp.Process(target=chess_worker, args=(child_conn, gamma)))
            processes[-1].start()
        
        games = 0
        done = []
        while games < args.epoch:
            for conn in conns:

                if conn.poll():
                    msg = conn.recv()

                    if msg['type'] == 'data':
                        model.train(msg['inp'], msg['rewards'])
                            
                        games += 1
                        if not args.quiet:
                            print('{}/{}'.format(games, args.epoch), end='\r')

                    elif msg['type'] == 'board':
                        conn.send(q_select(msg['boards'], model))

        for p in processes:
            p.terminate()
            p.join()

        gamma *= .9
        model.save()

    test_random(model, 10)
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
