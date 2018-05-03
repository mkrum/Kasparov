import chess
import chess.pgn
import chess.uci
import chess.xboard


stock = chess.uci.popen_engine('/Users/Dan/dev/nd/senior/ml/project/Stockfish/src/stockfish')
stock.uci()
stock.setoption({'Skill Level': 0})

sun = chess.xboard.popen_engine('/Users/Dan/dev/nd/senior/ml/project/sunfish/xboard.py')
sun.xboard()

board = chess.Board()


while not board.is_game_over(claim_draw=True):
    if board.turn:
        stock.position(board)
        move, _ = stock.go()
        board.push(move)
    else:
        sun.setboard(board)
        move = sun.go()
        board.push(move)

sun.exit()

print(board.result())
