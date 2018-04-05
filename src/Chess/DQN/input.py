import collections
import numpy as np
import chess

SIZE = 8
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


def get_input(boards, history):
    ''' returns the representation of the game state '''
    depth_size = 12 * history + 9
    inp = np.zeros((SIZE, SIZE, depth_size), dtype=np.int16)
    constants = get_constants(boards[-1])
    for i in range(SIZE):
        for j in range(SIZE):
            square = chess.square(i, j)
            inp[i][j] = get_depth(boards, square, constants, depth_size, history)

    return inp


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
    b = board if board.turn else board.mirror()
    p1_castling = [int(b.has_kingside_castling_rights(True)),
                   int(b.has_queenside_castling_rights(True))]
    p2_castling = [int(b.has_kingside_castling_rights(False)),
                   int(b.has_queenside_castling_rights(False))]
    constants.extend(p1_castling)
    constants.extend(p2_castling)
    constants.append(board.halfmove_clock)
    repetitions = get_repetitions(board)
    constants.extend(repetitions)

    return constants



def get_depth(boards, square, constants, depth_size, history):
    ''' get the 105 length vector for a given row and column '''
    depth = [0] * depth_size
    cur = boards[-1]

    for board in boards[-history:]:
        b = board if cur.turn else board.mirror()
        piece = b.piece_at(square)

        if not piece:
            depth.extend([0] * 12)
        else:
            depth.extend(ONEHOT[piece.symbol()])

    depth.extend(constants)

    return np.array(depth[-depth_size:])


def main():
    boards = []
    board = chess.Board()
    boards.append(board)
    inp = get_input(boards, 1)
    print(inp)
    print(inp.shape)


if __name__ == '__main__':
    main()
