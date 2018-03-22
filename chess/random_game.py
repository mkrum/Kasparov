import random
import chess

def play_game():
    board = chess.Board()

    while not board.is_game_over():
        move = random.choice(list(board.legal_moves))
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
