#
# custom agents added here
#

from pickletools import optimize
from turtle import forward
from game import Agent
from game import Directions
import random

import torch

class QFunction(torch.nn.Module):
    def __init__(self, action_space_size) -> None:
        super().__init__()
        self.action_space_size = action_space_size

        # neural network layers
        self.linear_1 = torch.nn.Linear(3, self.action_space_size)

    def forward(self, state):
        # calculate action probabilities
        input_data = torch.tensor([1,2,3], dtype=torch.float32)
        q_values = self.linear_1(input_data)
        #q_values = torch.nn.Softmax(q_values)

        return q_values

class CustomAgent(Agent):
    """
    A custom agent.
    """
    def __init__(self, index=0, alpha=0.01, epsilon=0, gamma=0.99):
        super().__init__(index)
        self.action_space = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.epsilon = epsilon
        self.gamma = gamma

        # define q function
        # define optimizer
        self.q_function = QFunction(len(self.action_space))
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=alpha)

        print(list(self.q_function.parameters()))

    def getAction(self, state):
        # get q-values
        # convert to probabilities
        q_values = self.q_function(state)
        #action_probabilities = torch.nn.functional.normalize(q_values)
        action_probabilities = q_values

        # get action
        # only allow for legal actions
        legal_actions = state.getLegalActions(self.index)
        action_mask = torch.tensor([x in legal_actions for x in self.action_space])
        action_probabilities = action_probabilities * action_mask
        action = self.action_space[torch.argmax(action_probabilities)]

        # use epsilon-greedy policy
        if(torch.rand((1,1)).item() < self.epsilon):
            # take random action
            action = legal_actions[torch.randint(0, len(legal_actions), (1,1)).item()]


        # calculate loss
        next_state = state.generateSuccessor(self.index, action)
        current_reward = torch.tensor(next_state.data.score - state.data.score)
        future_rewards = torch.mean(self.q_function(state) - self.gamma*self.q_function(next_state))
        total_reward = current_reward + future_rewards
        loss = -1 * total_reward

        # update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(loss)

        return action
