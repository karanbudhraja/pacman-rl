#
# custom agents added here
#

from turtle import forward
from game import Agent
from game import Directions
import random

import torch
import numpy as np

class QFunction(torch.nn.Module):
    def __init__(self, epsilon) -> None:
        super().__init__()
        self.action_space = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.epsilon = epsilon

        # neural network layers
        self.linear_1 = torch.nn.Linear(3, len(self.action_space))

    def forward(self, state, legal_actions):
        # calculate action probabilities
        action_probabilities = torch.rand((1, len(self.action_space)))

        # only allow for legal actions
        action_mask = torch.tensor([x in legal_actions for x in self.action_space])
        action_probabilities = action_probabilities * action_mask

        # select epsilon-greedy policy
        action = self.action_space[torch.argmax(action_probabilities)]
        if(torch.rand((1,1)).item() < self.epsilon):
            # take random action
            action = legal_actions[torch.randint(0, len(legal_actions), (1,1)).item()]

        return action

class CustomAgent(Agent):
    """
    A custom agent.
    """
    def __init__(self, index=0):
        super().__init__(index)

        # define q function
        self.q_function = QFunction(1)

    def getAction(self, state):
        # take action
        legal_actions = state.getLegalActions(self.index)
        action = self.q_function(state, legal_actions)

        # update policy
        next_state = state.generateSuccessor(self.index, action)
        pass


        x = state.data.score
        y = next_state.data.score
        print(x, y)

        return action
