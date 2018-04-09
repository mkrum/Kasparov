import numpy as np
import chess
import chess.pgn


def to_pgn(board):
    '''
    Converts a board object into a pgn file
    '''
    game = chess.pgn.Game()
    moves = list(board.move_stack)
    node = game.add_variation(moves[0])

    for i in range(1, len(moves)):
        node = node.add_variation(moves[i])

    return str(game)

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

