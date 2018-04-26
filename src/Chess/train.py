from __future__ import print_function

import os
import copy
import multiprocessing as mp
import argparse
import time

from shutil import copy2

import chess
import chess.uci
from util import get_input, to_pgn

def test_random(model, size, history):
    '''
    tests the model against a random opponent
    '''
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
                move = select(boards, model, history)
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


def main(args):

    model = DQN()
    if args.load is not None:
        print('Loading {}'.format(args.load))
        model.load(args.load)

    for it in range(args.iter):
        if it > 0:

            if not args.debug:
                f = open(args.path + '/' + str(it), 'w')
                f.write('Gamma: {}\n'.format(args.gamma))
                f.write('Average Loss: {}\n'.format(sum(losses) / len(losses)))
                f.write('Time (Minutes): {}\n'.format(((end - start) / 60)))
                f.write('\n\n')
                f.close()

        start = time.time()
        conns = []
        processes = []

        for p in range(args.threads):
            parent_conn, child_conn = mp.Pipe()
            conns.append(parent_conn)
            processes.append(mp.Process(target=chess_worker, args=(child_conn, args)))
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
                        conn.send(model.evaluate(np.expand_dims(get_simple_input(msg['boards'], args.history), 0)))

                    elif msg['type'] == 'end':

                        games += 1
                        if not args.quiet:
                            print('{}/{}'.format(games, args.epoch), end='\r')

        for p in processes:
            p.terminate()
            p.join()

        args.gamma *= args.decay

        if not args.debug:
            model.save(args.path + '/.modelprog')

        end = time.time()

    if not args.debug:
        model.save(args.path + '/.modelprog')
        f = open(args.path + '/final', 'w')
        f.write('Gamma: {}\n'.format(gamma))
        f.write('Average Loss: {}\n'.format(sum(losses) / len(losses)))
        f.write('\n\n')
        f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify the training setup.')

    parser.add_argument('--model', metavar='e', type=str, default='dqn',
                        help='Specify which model you want to train')

    parser.add_argument('--threads', metavar='t', type=int, default=20,
                        help='Number of threads.')

    parser.add_argument('--iter', metavar='i', type=int, default=100,
                        help='Number of iterations')

    parser.add_argument('--epoch', metavar='e', type=int, default=100,
                        help='Size of each epoch')

    parser.add_argument('--quiet', dest='quiet', action='store_const',
                        const=True, default=False, help='Repress progress output')

    parser.add_argument('--debug', dest='debug', action='store_const',
                        const=True, default=False, help='Repress any saving')

    parser.add_argument('--load', metavar='l', type=str, default=None,
                        help='Load a pre existing file')

    parser.add_argument('--gamma', metavar='g', type=float, default=1.0,
                        help='Exploration parameter')

    parser.add_argument('--lam', metavar='l', type=float, default=50,
                        help='Lambda value for the situtional distribution')

    parser.add_argument('--decay', metavar='d', type=float, default=0.9,
                        help='Exploration parameter')

    parser.add_argument('--history', metavar='h', type=int, default=2,
                        help='Number of previous boards to include in the input')

    savepath = 'res/' + '-'.join(time.ctime().split())
    parser.add_argument('--path', metavar='p', type=str, default=savepath,
                        help='Experiment path')

    arguments = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not arguments.debug:
        os.mkdir(arguments.path)
        settings_file = open(arguments.path + '/settings.txt', 'w')
        for arg in vars(arguments):
            settings_file.write('{} {}\n'.format(arg, getattr(arguments, arg)))

        settings_file.close()

    if arguments.model == 'td':
        from temporal import *
        copy2('temporal.py', arguments.path)
    elif arguments.model == 'dqn':
        from dqn import *
        copy2('dqn.py', arguments.path)
    elif arguments.model == 'alt_dqn':
        from dqn import *
        copy2('alt_dqn.py', arguments.path)
    elif arguments.model == 'app':
        from apprentice import *
        copy2('apprentice.py', arguments.path)
    elif arguments.model == 's_td':
        from sit_temporal import *
        copy2('sit_temporal.py', arguments.path)
    else:
        print('Model not found: {}'.format(arguments.model))
        exit()


    main(arguments)
