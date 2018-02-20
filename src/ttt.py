
import numpy as np
import operator
import copy

class TicTacToe(object):
    

    player_ids = [1, 2]
    action_dict = [ (0, i) for i in range(3)] + [ (1, i) for i in range(3)] \
        + [(2, i) for i in range(3) ] 
    
    def __init__(self):
        self.board = np.zeros((3, 3))

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


    def get_board(self, player_id):
        relative_board = copy.copy(self.board)

        relative_board[self.board == player_id] = 1
        relative_board[self.board != player_id] = -1
        relative_board[self.board == 0] = 0

        return relative_board

    def get_board_raw(self):
        return self.board

    def move(self,  player_id, move):
        self.board[self.action_dict[move]] = player_id

    def place(self, player_id, x, y):
        self.board[x][y] = player_id




