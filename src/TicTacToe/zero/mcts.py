import sys
sys.path.append('..')

from ttt import TicTacToe
import copy
import numpy as np

def softmax(vals):
    '''
    scales all values to [0, 1]
    '''
    return [ x / sum(vals) for x in vals]

def random_choice(board):
    '''
    Random selection algorithm
    '''

    possible = np.where(board == 0)
    choice = np.random.random_integers(low=0, high=(possible[0].size - 1), size=1)

    return (possible[0][choice[0]], possible[1][choice[0]])

class TestModel(object):
    '''
    Simple model placeholder
    '''

    def evaluate(self, game, player):
        '''
        returns the value of a board position
        '''

        prob = np.ones((3, 3)) #uniform prob, placeholder
        winner = game.check_win()

        if winner is None:
            return (prob, 0)
        elif winner == 0:
            return (prob, -1)

        return (prob, 1) if winner == player else (prob, -1)

class MCTS(object):

    def __init__(self, model, game, current_player):
        self.model = model
        self.root = Node(self.model, game, current_player)

    def search(self, iters=300):
        '''
        Perform the MCTS search algorithm from the root of the tree
        '''

        for _ in range(iters):
            self.root.search(self.model)
        

        return self.root.select_move()
    
    def get_probabilities(self, tau):
        '''
        returns the probabilites found in the MCTS for each possible move
        '''

        prob_board = np.zeros((1, 3, 3, 1))
        vals = [ (e.move, e.N ** (1/tau)) for e in self.root.edges ]
        for move, prob in vals:
            x, y = move
            prob_board[0, x, y, 0] = prob

        return prob_board

class Edge(object):

    def __init__(self, parent, child, move, prior):
        self.parent = parent
        self.child = child
        self.move = move

        self.P = prior
        self.N = 0
        self.Q = child.value
        self.U = 0
        
    def backprop(self, value):
        '''
        Adjust the values of all the edges in the current path based on the new node
        '''

        if self.parent.leading_edge is not None:
            self.parent.leading_edge.backprop(-1* value)

        self.N += 1
        self.Q += 1.0 /self.N * (value - self.Q)
        self.U = self.P / (1.0 + self.N)


class Node(object):

    def __init__(self, model, game, player):
        self.model = model
        self.game = game
        self.leading_edge = None
        self.edges = []
        self.probs, self.value = model.evaluate(game, (player % 2) + 1)
        self.player = player

    def attach(self, edge):
        self.leading_edge = edge

    def is_leaf(self):
        return len(self.edges) == 0
    
    def is_terminal(self):
        possible_moves = self.game.get_possible()
        return len(possible_moves) == 0

    def search(self, model):
        '''
        Keep selecting nodes in the tree until you reach a leaf. Then, expand the node, select one of its children
        and backpropagate its value back to the root node.
        '''

        if self.is_leaf():
            self.expand()
            edge = self.select_edge()

            if edge:
                edge.backprop(edge.child.value)

            if len(self.edges) == 0:
                self.leading_edge.backprop(self.value)

        else:
            edge = self.select_edge()
            edge.child.search(model)
    
    def expand(self):
        '''
        Create children for all the possible moves from a node
        '''

        possible_moves = self.game.get_possible()
        edge_probs = []
        for x, y in possible_moves:
            edge_probs.append(self.probs[0, x, y])

        edge_probs = softmax(edge_probs)
        
        for move, prob in zip(possible_moves, edge_probs):
            hyp_game = copy.deepcopy(self.game)     
            hyp_game.place(self.player, move[0], move[1])
            
            child_node = Node(self.model, hyp_game, (self.player % 2) + 1)
            new_edge = Edge(self, child_node, move, prob)
                
            child_node.attach(new_edge)

            self.edges.append(new_edge)
    
    def select_edge(self):
        '''
        Node selection algorithm, based off the "upper confidence bound" of the value estimate
        '''

        if len(self.edges) == 0:
            return None

        uct = list(map(lambda e: e.U + e.Q, self.edges))
        return self.edges[uct.index(max(uct))]

    def select_move(self):
        '''
        selects the best move from the root node, decided by total number of visits
        '''

        edge_counts = list(map(lambda e: e.N, self.edges))
        
        return self.edges[edge_counts.index(max(edge_counts))].move

def evaluate(model, game, next_player, tau=1.0):
    '''
    Board evaluation function using MCTS
    '''
    tree = MCTS(model, copy.deepcopy(game), next_player)
    x, y = tree.search()
    prob = tree.get_probabilities(1.0)
    return x, y, prob

if __name__ == '__main__':

    wins = {0: 0, 1: 0, 2: 0}
    model = TestModel() 
    boards = []
    for _ in range(100):
        game = TicTacToe()
        
        pid = np.random.random_integers(low=1, high=2, size=1)[0]
        winner = None
        while winner is None:

            board = game.get_board(pid)
            boards.append(copy.copy(game.board))

            if pid == 1:
                x, y = random_choice(board)
            else:
                test = MCTS(model, copy.deepcopy(game), pid)
                x, y = test.search()

            game.place(pid, x, y)

            winner = game.check_win()

            pid = (pid % 2) + 1
        wins[winner] += 1
        if winner == 0:

            for board in boards:
                print(board)
        
        boards = []
            
    print('Wins: %d Ties: %d Losses: %d' % (wins[2], wins[0], wins[1]))


