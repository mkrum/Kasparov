from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import os
import copy
import multiprocessing as mp
import argparse
import time
import random
import matplotlib.pyplot as plt

from shutil import copy2


def test_random(model):
    
    wins = 0
    ties = 0
    losses = 0
    
    for _ in range(500):
        game = TicTacToe()

        player_pid = 2 - int(random.random() >= 0.5)
        print(player_pid)

        pid = 1
        boards = [copy.deepcopy(game)]
        i = 0

        print(game.board)
        while game.check_win() is None:
            
            if pid == player_pid:
                move = select(boards, model, 2)
            else:
                move = random.choice(game.legal_moves)

            game = game.push(move)
            boards.append(copy.deepcopy(game))
            print(game.board)

            pid = (pid % 2 + 1)

        res = game.check_win()
        print(res)

        if res == player_pid:
            wins += 1
        elif res == 0:
            ties += 1
        else:
            losses += 1

    print('{} {} {}'.format(wins, ties, losses))
    return wins, ties, losses


def main(args):

    model = DQN()
    if args.load is not None:
        print('Loading {}'.format(args.load))
        model.load(args.load)
    
    win_rate = []
    
    w, t, l = test_random(model)

    win_rate.append(w / float(sum([w, t, l])))
    print(w / float(sum([w, t, l])))


    for it in range(args.iter):
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
                        print(games, end='\r')
                        games += 1

        for p in processes:
            p.terminate()
            p.join()

        w, t, l = test_random(model)

        win_rate.append(w / float(sum([w, t, l])))
        print(w / float(sum([w, t, l])))

        args.gamma *= args.decay
        end = time.time()

    plt.plot(range(len(win_rate)), win_rate)
    plt.xlabel('Epoch')
    plt.ylabel('Win Rate')
    plt.savefig('-'.join(time.ctime().split()))


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

    if arguments.model == 'td':
        from temporal import *
        copy2('temporal.py', arguments.path)
    elif arguments.model == 'dqn':
        from dqn import *
        copy2('dqn.py', arguments.path)
    elif arguments.model == 'alt_dqn':
        from alt_dqn import *
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
