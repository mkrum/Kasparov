
import numpy as np
import operator
import copy

from util import get_simple_input

class TicTacToe(object):
    
    def __init__(self, turn=1, board=np.zeros((3,3))):
        self.turn = turn
        self.board = board
        self.legal_moves = self.get_possible()

    def check_win(self):

        for i in range(3):
            row = np.unique(self.board[:, i])

            if row.size == 1 and row[0] > 0:
                return row[0]

            col = np.unique(self.board[i, :])

            if col.size == 1 and col[0] > 0:
                return col[0]
        
        diag_one = np.unique(np.array([self.board[i][i] for i in range(3)]))
        diag_two = np.unique(np.array([self.board[i][2 - i] for i in range(3)]))

        if diag_one.size == 1 and diag_one[0] > 0:
            return diag_one[0]

        if diag_two.size == 1 and diag_two[0] > 0:
            return diag_two[0]
        
        if self.board[self.board == 0].size == 0:
            return 0

        return None
    
    def at(self, x, y):
        return self.board[x][y]

    def get_board(self, player_id, curr_board=None):
        if curr_board is None:
            curr_board = self.board

        relative_board = copy.copy(curr_board)

        relative_board[curr_board == player_id] = 1
        relative_board[curr_board != player_id] = -1
        relative_board[curr_board == 0] = 0

        return relative_board

    def push(self, move):
        '''
        register a move on the board
        '''
        x, y = move
        new_board = copy.deepcopy(self.board)
        new_board[x][y] = self.turn

        return TicTacToe(turn=(self.turn % 2 + 1), board=new_board)

    def get_possible(self):
        '''
        returns the possible moves for the current game state
        '''
        if self.check_win() is not None:
            return []

        possible = np.where(self.board == 0)
        return [ (possible[0][i], possible[1][i]) for i in range(len(possible[0]))]


if __name__ == '__main__':
    game = TicTacToe()
    boards = [copy.deepcopy(game)]
    while game.check_win() is None:
        possible_moves = game.legal_moves
        print(game.board)
        print(possible_moves)
        move_ind = int(input('Select: '))
        move = possible_moves[move_ind]

        game = game.push(move)
        boards.append(game)
    
    for b in boards:
        print(b.board)
        
    for i in range(1, len(boards) + 1):
        inp = get_simple_input(boards[:i], 2)
        print(inp[:, :, 0])
        print(inp[:, :, 1])
        print()


