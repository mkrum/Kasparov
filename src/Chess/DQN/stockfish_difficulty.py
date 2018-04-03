import chess
import chess.uci

def play_game():
    board = chess.Board()

    engine1 = chess.uci.popen_engine('/Users/Dan/dev/nd/senior/ml/project/Stockfish/src/stockfish')
    engine1.setoption({'Skill Level': 0})

    engine2 = chess.uci.popen_engine('/Users/Dan/dev/nd/senior/ml/project/Stockfish/src/stockfish')
    engine2.setoption({'Skill Level': 20})

    while not board.is_game_over(claim_draw=True):
        if board.turn:
            engine = engine1
        else:
            engine = engine2

        engine.position(board)
        move, _ = engine.go()
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
