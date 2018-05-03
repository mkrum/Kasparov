import random
import collections
import numpy as np

SIZE = 3

def board_value(board, player):
    #based on hans berliner

    state = np.zeros((3, 3))

    for i in range(SIZE):
        for j in range(SIZE):
            if board.at(i, j) == player:
                state[i][j] = 1
            elif board.at(i, j) != 0:
                state[i][j] = -1

    return state


def get_simple_input(boards, history):
    ''' returns the representation of the game state '''
    boards = boards[-history:]
    cur = boards[0] 
    
    player = cur.turn
    inp = np.zeros((SIZE, SIZE, 2))
    for i, board in enumerate(boards):
        inp[:, :, i] = board_value(board, player)

    return inp


def build_input(boards, rewards, history):
    ''' Converts a list of boards through time into respective model inputs '''
    size = len(boards) - 1
    inputs = np.zeros((size, SIZE, SIZE, history))

    for i in range(2, len(boards) + 1):
        inputs[(i - 2), :, :, :] = get_simple_input(boards[:i], history)

    paritions = np.ceil(float(size)/10.0)
    return np.array_split(inputs, paritions), np.array_split(rewards, paritions)




if __name__ == '__main__':
    main()
