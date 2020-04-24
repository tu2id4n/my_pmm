'''An agent that preforms a random action each step'''
from pommerman.agents import BaseAgent
import random


class RandAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        return random.randint(0, 120)
        # return 121