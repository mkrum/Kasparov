
from ttt import TicTacToe
import numpy as np
import random
import copy
from math import sqrt, log


def get_possible_moves(board):
    '''
    Generates all possible tic tac toe moves
    '''
    possible = np.where(board == 0)
    return [ (possible[0][i], possible[1][i]) for i in range(len(possible[0]))]

def check_win(board):
    '''
    Checks if the board is a terminal state
    '''
    for i in range(3):
        row = np.unique(board[:, i])

        if row.size == 1 and row[0] != 0:
            return row[0]

        col = np.unique(board[i, :])

        if col.size == 1 and col[0] != 0:
            return col[0]
    
    diag_one = np.unique(np.array([board[i][i] for i in range(3)]))
    diag_two = np.unique(np.array([board[i][2 - i] for i in range(3)]))

    if diag_one.size == 1 and diag_one[0] != 0:
        return diag_one[0]

    if diag_two.size == 1 and diag_two[0] != 0:
        return diag_two[0]
    
    if board[board == 0].size == 0:
        return 0

    return None

class Node(object):
    
    wins = 0
    visits = 0

    def __init__(self, board, next_player, parent=None, move=None):
        self.move = move
        self.parent = parent
        self.board = board
        self.next_player = next_player
        self.owner = (next_player % 2) + 1

        self.generate_children(next_player)

    def generate_children(self, pid):
        ''' Generate all first degree children of a 
            board
        '''
        moves = get_possible_moves(self.board)
       
        self.children = []
        for x, y in moves:
            new_board = copy.copy(self.board)
            new_board[x, y] = pid

            self.children.append(Node(new_board, self.owner, self, (x, y)))
    
    def backprop(self, winner):
        '''
        Update visit and win counts for previous nodes
        '''
        if winner is not None:
            self.visits += 1

            if winner == self.owner:
                self.wins += 1

            if self.parent:
                self.parent.backprop(winner)
        else:
            winner = check_win(self.board)
            if winner:
                self.backprop(winner)
    
    def rollout(self):
        winner = check_win(self.board)

        if winner is not None:
            self.backprop(winner)
        else:
            unexplored_children = [ c for c in self.children if c.visits == 0 ]
            
            if unexplored_children:
                np.random.choice(unexplored_children).random_rollout()
            else:
                next_node = self.uct_select()
                next_node.rollout()

    def uct_select(self):
        total_visits = sum([ c.visits for c in self.children ])
        uct_score = list(map(lambda child: child.wins / child.visits + sqrt(2*log(total_visits) / child.visits), 
                            self.children))
        return self.children[uct_score.index(max(uct_score))]

    def next_move(self):
        visits = [ c.visits for c in self.children ]
        return self.children[ visits.index(max(visits))].move


def evaluate(board, player):
    '''
    Board evaluation function using MCTS
    '''
    tree = Node(copy.copy(board), player)
     
    for _ in range(100):
        tree.rollout()

    return tree.next_move()

if __name__ == '__main__':
    test_board = np.zeros((3, 3))
    test_board[0][0] = 2
    test_board[0][1] = 2

    print(evaluate(test_board, 1))
