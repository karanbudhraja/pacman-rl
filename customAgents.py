#
# custom agents added here
#

from pickletools import optimize
from turtle import forward

from cairo import STATUS_FILE_NOT_FOUND
from game import Agent
from game import Directions

import torch

class ValueFunction(torch.nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()

        # neural network layers
        self.convolution_1 = torch.nn.Conv2d(input_size[0], input_size[0], 3)
        self.pooling_1 = torch.nn.MaxPool2d(2)
        self.convolution_2 = torch.nn.Conv2d(input_size[0], input_size[0], 2)
        self.pooling_2 = torch.nn.MaxPool2d((1,2))
        self.linear_1 = torch.nn.Linear(36, 6)
        self.linear_2 = torch.nn.Linear(6, 1)

    def forward(self, state):
        # calculate action probabilities
        value = self.convolution_1(state)
        value = self.pooling_1(value)
        value = self.convolution_2(value)
        value = self.pooling_2(value)
        value = torch.flatten(value)
        value = self.linear_1(value)
        value = self.linear_2(value)
        value = torch.tanh(value)

        return value

class PolicyFunction(torch.nn.Module):
    def __init__(self, input_size, action_space_size) -> None:
        super().__init__()
        self.action_space_size = action_space_size

        # neural network layers
        self.convolution_1 = torch.nn.Conv2d(input_size[0], input_size[0], 3)
        self.pooling_1 = torch.nn.MaxPool2d(2)
        self.convolution_2 = torch.nn.Conv2d(input_size[0], input_size[0], 2)
        self.pooling_2 = torch.nn.MaxPool2d((1,2))
        self.linear_1 = torch.nn.Linear(36, self.action_space_size)

    def forward(self, state):
        # calculate action probabilities
        policy = self.convolution_1(state)
        policy = self.pooling_1(policy)
        policy = self.convolution_2(policy)
        policy = self.pooling_2(policy)
        policy = torch.flatten(policy)
        policy = self.linear_1(policy)
        policy = torch.nn.functional.softmax(policy, dim=-1)

        return policy

class CustomAgent(Agent):
    """
    A custom agent.
    """
    def __init__(self, index=0, alpha=0.001, epsilon=0, gamma=0.99):
        super().__init__(index)
        self.action_space = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.epsilon = epsilon
        self.gamma = gamma
        self.data_buffer = []
        self.data_buffer_limit = 100
        self.previous_score = 0

        # define q function
        # define optimizer
        self.state_symbols = ['o', ' ', '%', 'G', '>', '.', 'v', '<', '^']
        self.grid_size = (7, 20)
        input_size = [len(self.state_symbols), *self.grid_size]
        self.value_function = ValueFunction(input_size)
        self.policy_function = PolicyFunction(input_size, len(self.action_space))
        self.optimizer = torch.optim.Adam(self.value_function.parameters(), lr=alpha)

        # logging
        self.log_file_name = "loss.txt"
        with open(self.log_file_name, "a") as log_file:
            log_file.truncate()

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

    def update(self):
        # perform gradient descent
        empirical_values = []
        estimated_values = []
        for index in torch.arange(1, len(self.data_buffer)):
            previous_index = index - 1
            [previous_state_tensor, _] = self.data_buffer[previous_index]
            [state_tensor, current_reward] = self.data_buffer[index]
            empirical_state_value = torch.tanh(current_reward)
            estimated_state_value = self.gamma*self.value_function(state_tensor) - self.value_function(previous_state_tensor)
            empirical_values.append(empirical_state_value)
            estimated_values.append(estimated_state_value)
        loss_function = torch.nn.MSELoss()
        loss = loss_function(torch.tensor(empirical_values, dtype=torch.float32, requires_grad=True), torch.tensor(estimated_values, dtype=torch.float32, requires_grad=True))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def getAction(self, state):
        #
        # compute action
        #

        # get legal actions
        legal_actions = state.getLegalActions(self.index)

        # get next possible states
        # get state values
        legal_successor_values = []
        for legal_action in legal_actions:
            legal_successor_state = state.generateSuccessor(self.index, legal_action)
            legal_successor_value = self.value_function(self.state_to_tensor(legal_successor_state))
            legal_successor_values.append(legal_successor_value)
        legal_successor_values = torch.cat(legal_successor_values)
        action = legal_actions[torch.argmax(legal_successor_values)]
        
        # use epsilon-greedy policy
        if(torch.rand((1,1)).item() < self.epsilon):
            # take random action
            action = legal_actions[torch.randint(0, len(legal_actions), (1,1)).item()]

        #
        # manage data buffer
        #

        # convert state
        # add data to buffer
        successor_state = state.generateSuccessor(self.index, action)
        successor_reward = torch.tanh(torch.tensor(successor_state.data.score - state.data.score))
        self.data_buffer.append([self.state_to_tensor(successor_state), successor_reward])

        # update model when buffer is full
        if(len(self.data_buffer) == self.data_buffer_limit):
            loss = self.update()
            print(loss)
            with open(self.log_file_name, "a") as log_file:
                log_file.write(str(loss.item()) + "\n")

            # empty buffer
            self.data_buffer = []

        return action
