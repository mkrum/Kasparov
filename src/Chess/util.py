import collections
import numpy as np
import chess

SIZE = 8
DEPTH_SIZE = 105
STEP = 8

# encoding for a square that contains a piece
ONEHOT = {
    'P': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'p': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'n': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'b': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'k': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }


def get_input(boards):
    ''' return the 8x8x105 representation of the game state '''
    inp = np.zeros((SIZE, SIZE, DEPTH_SIZE), dtype=np.int16)
    constants = get_constants(boards[-1])
    for i in range(SIZE):
        for j in range(SIZE):
            square = chess.square(i, j)
            inp[i][j] = get_depth(boards, square, constants)

    return np.expand_dims(inp, 0)


def get_repetitions(board):
    ''' get consecutive repetitions.
        code from  can_claim_threefold_repetition function of python-chess package
    '''
    transposition_key = board._transposition_key()
    transpositions = collections.Counter()
    transpositions.update((transposition_key, ))

    # Count positions.
    switchyard = collections.deque()
    while board.move_stack:
        move = board.pop()
        switchyard.append(move)

        if board.is_irreversible(move):
            break

        transpositions.update((board._transposition_key(), ))

    while switchyard:
        board.push(switchyard.pop())

    count = transpositions[transposition_key]
    reps = [0, 0]

    if count == 1:
        reps = [1, 0]

    if count == 2:
        reps = [0, 1]

    return reps


def get_constants(board):
    '''color, total move count, castling, no-progress count, and repetitions'''
    constants = [board.turn, board.fullmove_number]
    p1_castling = [0, 0]
    p2_castling = [0, 0]
    constants.extend(p1_castling)
    constants.extend(p2_castling)
    constants.append(board.halfmove_clock)
    repetitions = get_repetitions(board)
    constants.extend(repetitions)

    return constants


def get_depth(boards, square, constants):
    ''' get the 105 length vector for a given row and column '''
    depth = [0] * DEPTH_SIZE
    cur = boards[-1]

    for board in boards:
        b = board if cur.turn else board.mirror()
        piece = b.piece_at(square)

        if not piece:
            depth.extend([0] * 12)
        else:
            depth.extend(ONEHOT[piece.symbol()])

    depth.extend(constants)

    return np.array(depth[-DEPTH_SIZE:])


def main():
    boards = []
    board = chess.Board()
    boards.append(board)
    #inp = get_input(board.move_stack)
    print(board.move_stack)
    print(inp)


if __name__ == '__main__':
    main()
