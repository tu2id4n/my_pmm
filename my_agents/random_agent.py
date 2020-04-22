'''An agent that preforms a random action each step'''
from pommerman.agents import BaseAgent
import random


class RandAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""
    def __init__(self, *args, **kwargs):
        super(RandAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        return random.randint(0, 120)
