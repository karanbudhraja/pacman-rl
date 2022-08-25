#
# custom agents added here
#

from pickletools import optimize
from turtle import forward

from cairo import STATUS_FILE_NOT_FOUND
from game import Agent
from game import Directions
import random

import torch

import numpy as np

class QFunction(torch.nn.Module):
    def __init__(self, input_size, action_space_size) -> None:
        super().__init__()
        self.state_symbols = ['o', ' ', '%', 'G', '>', '.', 'v', '<', '^']
        self.action_space_size = action_space_size

        # neural network layers
        self.linear_1 = torch.nn.Linear(3, self.action_space_size)

    def forward(self, state):
        # calculate action probabilities
        input_data = torch.tensor([1,2,3], dtype=torch.float32)
        q_values = self.linear_1(input_data)
        #q_values = torch.nn.functional.softmax(q_values)
        
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
        self.data_buffer = []

        # define q function
        # define optimizer
        self.q_function = QFunction(len(self.action_space))
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=alpha)

        print(list(self.q_function.parameters()))

    def state_to_tensor(self, state):
        # get state data without score
        # convert to matrix
        state_information_string_data = str(state.data).split("\n")[:-2]
        symbol_information_numeric_data = []
        for symbol in self.state_symbols:
            state_information_numeric_data = []
            for row in state_information_string_data:
                state_information_numeric_data.append(torch.tensor([item == symbol for item in row]))
            state_information_numeric_data = torch.tensor(torch.cat(state_information_numeric_data).reshape(len(state_information_string_data), -1), dtype=torch.float32)
            symbol_information_numeric_data.append(state_information_numeric_data)
        symbol_information_numeric_data = torch.cat(symbol_information_numeric_data).reshape(len(self.state_symbols), len(state_information_string_data), -1)

        return symbol_information_numeric_data

    def getAction(self, state):

        # # get action
        # # only allow for legal actions
        # legal_actions = state.getLegalActions(self.index)
        # action_mask = torch.tensor([x in legal_actions for x in self.action_space])
        # action_probabilities = action_probabilities * action_mask
        # action = self.action_space[torch.argmax(action_probabilities)]

        # # use epsilon-greedy policy
        # if(torch.rand((1,1)).item() < self.epsilon):
        #     # take random action
        #     action = legal_actions[torch.randint(0, len(legal_actions), (1,1)).item()]

        # # calculate loss
        # next_state = state.generateSuccessor(self.index, action)
        # current_reward = torch.tensor(next_state.data.score - state.data.score)
        # future_rewards = torch.mean(self.q_function(state) - self.gamma*self.q_function(next_state))
        # total_reward = current_reward + future_rewards
        # loss = -1 * total_reward

        # # update policy
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # print(loss)

        # get q-values
        state_tensor = self.tate_to_tensor(state)
        action_probabilities = self.q_function(state)
        

        print(state)


        legal_actions = state.getLegalActions(self.index)
        action = legal_actions[torch.randint(0, len(legal_actions), (1,1)).item()]

        return action
