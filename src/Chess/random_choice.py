

import chess
import numpy as np

board = chess.Board()


for move in board.legal_moves:
    print(move)

