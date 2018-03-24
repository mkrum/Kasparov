
import numpy as np
import operator
import copy

class TicTacToe(object):
    
    def __init__(self, history_size=0):
        self.history = []
        self.board = np.zeros((3, 3))

        for _ in range(history_size):
            self.history.append(np.zeros((3, 3)))
    
    def get_input(self, player_id):
        '''
        returns the board representation used for the network
        '''
        T = len(self.history)
        M = 2
        L = 1

        input_planes = np.zeros((3, 3, (T * M + L)))

        #L plane
        if player_id == 1:
            input_planes[:, :, 0] = np.zeros((3, 3))
        else:
            input_planes[:, :, 0] = np.ones((3, 3))
        
        relevant_history = []
        for h in self.history:
            rel_h = self.get_board(player_id, h)
            relevant_history.append(rel_h)

        for i in range(T):
            input_planes[:, :, i + 1:i + 3] = self.to_plane(relevant_history[i])

        input_planes = np.expand_dims(input_planes, 0)
        return input_planes
    
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


    def get_board(self, player_id, curr_board=None):
        if curr_board is None:
            curr_board = self.board

        relative_board = copy.copy(curr_board)

        relative_board[curr_board == player_id] = 1
        relative_board[curr_board != player_id] = -1
        relative_board[curr_board == 0] = 0

        return relative_board

    def get_board_raw(self):
        return self.board

    def to_plane(self, board):
        '''
        converts a board to its planar representation
        '''

        planes = np.zeros((3, 3, 2))
        
        for i in range(3):
            for j in range(3):
                if board[i, j] == 1:
                    planes[i, j, 0] = 1

                if board[i, j] == -1:
                    planes[i, j, 1] = 1

        return planes 

    def place(self, player_id, x, y):
        '''
        register a move on the board
        '''

        self.board[x][y] = player_id
        
        if len(self.history) > 0:
            self.history.pop(0) 
            self.history.append(self.board)

    def get_possible(self):
        '''
        returns the possible moves for the current game state
        '''
        if self.check_win() is not None:
            return []

        possible = np.where(self.board == 0)
        return [ (possible[0][i], possible[1][i]) for i in range(len(possible[0]))]
