from __future__ import print_function

import chess
import copy
import threading
import multiprocessing as mp
import numpy as np
import argparse
import sys
import time

import chess.uci
import os
from util import get_input
from chess_util import test_random, to_pgn


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
            processes.append(mp.Process(target=chess_worker, args=(child_conn, gamma, args.history)))
            processes[-1].start()
        
        games = 0
        losses = []
        while games < args.epoch:
            for conn in conns:

                if conn.poll():
                    msg = conn.recv()

                    if msg['type'] == 'data':
                        losses.append(model.train(msg['inp'], msg['rewards']))
                            
                    elif msg['type'] == 'board':
                        conn.send(select(msg['boards'], model, args.history))

                    elif msg['type'] == 'eval':
                        conn.send(model.evaluate(get_input(msg['boards'], args.history)))

                    elif msg['type'] == 'end':
                        games += 1
                        if not args.quiet:
                            print('{}/{}'.format(games, args.epoch), end='\r')

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

    parser.add_argument('--model', metavar='e', type=str, default='dqn',
                    help='Specify which model you want to train')

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

    parser.add_argument('--history', metavar='e', type=int, default=8,
                    help='Exploration parameter')

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    if args.model == 'td':
        from temporal import *
    elif args.model == 'dqn':
        from dqn import *
    elif args.model == 'app':
        from apprentice import *
    else:
        print('Model not found: {}'.format(args.model))
        exit()
        
    main(args)
