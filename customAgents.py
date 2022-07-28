#
# custom agents added here
#

from game import Agent
from game import Directions
import random

class CustomAgent(Agent):
    """
    A custom agent.
    """
    def getAction(self, state):
        legal = state.getLegalActions(self.index)
        move = random.choice(legal)

        return move
