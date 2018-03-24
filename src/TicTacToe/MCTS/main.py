import sys
sys.path.append('..')

from model import *

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
            x, y = random_choice(game.board)
        else:
            test = MCTS(model, copy.deepcopy(game), pid)
            x, y = test.search()

        game.place(pid, x, y)

        winner = game.check_win()

        pid = (pid % 2) + 1
    wins[winner] += 1

    if winner == -1:

        for board in boards:
            print(board)
    
    boards = []
        
print('Wins: %d Ties: %d Losses: %d' % (wins[2], wins[0], wins[1]))

