
import random

class DQN(object):

    def __init__(self):
        self.Q = {}
        self.alpha = .05
    
    def evaluate(self, boards):

        for b in boards:
            try:
                return self.Q[str(b)]
            except KeyError:
                return random.random()
    def train(self, boards, rewards):
    
        for ind, board in enumerate(boards):
            reward = rewards[ind]
            
            try:
                self.Q[str(board)] = (1 - self.alpha) * self.Q[str(board)] + self.alpha * (reward)
            except KeyError:
                self.Q[str(board)] = reward

