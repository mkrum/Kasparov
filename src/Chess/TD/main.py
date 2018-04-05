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
from util import get_input, to_pgn
import numpy as np
from model import DQN


N = 8

def build_input(boards):
    size = len(boards)
    inputs = np.zeros((size, 8, 8, 105))

    for i in range(1, len(boards) + 1):
        inputs[(i - 1), :, :, :] = get_input(boards[:i], N)

    return inputs

def q_select(board, model):
    '''
    Picks the worst board for the opponent 
    '''
    
    possible = np.where(board == 0)
    
    min_val = float('inf')
    for x, y in zip(possible[0], possible[1]):
        t_board = copy.copy(board)
        t_board *= -1
        t_board[x, y] = -1.0

        val = model.evaluate(t_board.reshape(1, 3, 3))

        if val < min_val:
            min_val = val
            min_x = x
            min_y = y

    return min_x, min_y

def q_select(boards, model):
    '''
    Simple selection algorithm

    Picks the move on the baord with the highest estimated value 
    '''
    if len(boards) == 0:
        boards.append(chess.Board())
    
    curr_board = boards[-1]
    possible = list(curr_board.legal_moves)
    
    min_val = float('inf')
    for move in possible:
        t_boards = copy.deepcopy(boards)
        t_curr = copy.copy(t_boards[-1])
        t_curr.push(move)
        t_boards.append(t_curr)

        val = model.evaluate(get_input(t_boards, N))
        if val < min_val:
            min_val = val
            best_move = move

    return best_move

def test_random(model, size=10):
    wins = 0
    draws = 0

    pgn = []
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

        pgn.append(to_pgn(board))

    print('Wins: {} Draws: {} Losses: {}'.format(wins, draws, size - (wins + draws)))
    return pgn

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
    lam = 0.7
    while True:
        board = chess.Board()
        player = 1

        rel_boards = {1: [], 2: []}
        vals = {1: [], 2: []}
        rewards = {1: [], 2: []}
        boards = []

        pid = 1
        while not board.is_game_over(claim_draw=True):

            if random.random() > gamma:
                connection.send({'type': 'board', 'boards': boards[-8:]})
                move = connection.recv()
            else:
                move = random.choice(list(board.legal_moves))

            rel_boards[pid].append(board)
            
            connection.send({'type': 'eval', 'boards': boards[-8:]})
            val = connection.recv()

            vals[pid].append(val)
            rewards[pid].append(0)

            boards.append(copy.copy(board))

            board.push(move)

            if len(boards) > 8:
                boards.pop(0)

            pid = (pid % 2 + 1)

        res = board.result()

        if res == '1-0':
            rewards[1][-1] = 1
            rewards[2][-1] = -1
        elif res == '0-1':
            rewards[1][-1] = -1
            rewards[2][-1] = 1

        targets = {1: [], 2: []}

        for i in range(len(rewards[1]) - 1):
            targets[1].append(rewards[1][i] + lam * vals[1][i + 1])
        targets[1].append(rewards[1][-1])

        for i in range(len(rewards[2]) - 1):
            targets[2].append(rewards[2][i] + lam * vals[2][i + 1])
        targets[2].append(rewards[2][-1])

        inp = build_input(rel_boards[1] + rel_boards[2])
        targets = np.expand_dims(np.array(targets[1] + targets[2]), 1)

        connection.send({'type': 'data', 'inp': inp, 'rewards': targets})    


def main(args):
    model = DQN()
    if args.load is not None:
        print('Loading {}'.format(args.load))
        model.load(args.load)

    gamma = args.gamma
    for it in range(args.iter):
        if it > 0:
            f = open(str(it), 'w')
            f.write('Gamma: {}\n'.format(gamma))
            f.write('Average Loss: {}\n'.format(sum(losses) / len(losses)))
            f.write('Time (Minutes): {}\n'.format(((end - start) / 60)))
            f.write('\n\n')
            pgn = test_random(model, 2)
            for g in pgn:
                f.write(g)
                f.write('\n\n')

            f.close()

        start = time.time()
        conns = []
        processes = []

        for p in range(args.threads):
            parent_conn, child_conn = mp.Pipe()
            conns.append(parent_conn)
            processes.append(mp.Process(target=chess_worker, args=(child_conn, gamma)))
            processes[-1].start()
        
        games = 0
        losses = []
        while games < args.epoch:
            for conn in conns:

                if conn.poll():
                    msg = conn.recv()

                    if msg['type'] == 'data':
                        losses.append(model.train(msg['inp'], msg['rewards']))
                            
                        games += 1
                        if not args.quiet:
                            print('{}/{}'.format(games, args.epoch), end='\r')

                    elif msg['type'] == 'board':
                        conn.send(q_select(msg['boards'], model))

                    elif msg['type'] == 'eval':
                        conn.send(model.evaluate(get_input(msg['boards'], N)))


        for p in processes:
            p.terminate()
            p.join()

        gamma *= .9
        model.save()
        end = time.time()

    model.save()
    f = open('final', 'w')
    f.write('Gamma: {}\n'.format(gamma))
    f.write('Average Loss: {}\n'.format(sum(losses) / len(losses)))
    f.write('\n\n')
    pgn = test_random(model, 2)
    for g in pgn:
        f.write(g)
        f.write('\n\n')

    f.close()


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
