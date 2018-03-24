import chess

b = chess.Board()
print(b.piece_at(chess.square(3, 3)))
print(b.piece_at(chess.square(0, 0)))
print(b.turn)
move = list(b.legal_moves)[0]
b.push(move)
print(b)
print(b.turn)
print(b.mirror().turn)
print(b.mirror())
