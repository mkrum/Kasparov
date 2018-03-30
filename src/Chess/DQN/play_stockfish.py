import random
import chess

import chess.uci
import os

engine = chess.uci.popen_engine(os.environ['SFSH'])
engine.uci()

def play_game():
    board = chess.Board()
    player = 1
    while not board.is_game_over(claim_draw=True):
        if player == 1:
            move = random.choice(list(board.legal_moves))
        else:
            engine.position(board)
            move, _ = engine.go(movetime=2000)
        player = (player % 2) + 1

        board.push(move)

    res = board.result()

    if res == '1-0':
        print('White wins')
    elif res == '0-1':
        print('Black wins')
    else:
        print('Draw')

    print('Move pairs: {}'.format(board.fullmove_number))

    print('Final board:')
    print(board)


def main():
    play_game()

if __name__ == '__main__':
    main()
    engine.quit()
