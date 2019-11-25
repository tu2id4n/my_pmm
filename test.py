import pommerman
from pommerman import agents
from pommerman import constants
from my_agents import *

agent_list = [
    agents.DockerAgent('multiagentlearning/hakozakijunctions', port=12345),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent()
]
env = pommerman.make('PommeTeamCompetition-v1', agent_list)

for episode in range(10000):
    obs = env.reset()
    done = False
    while not done:
        all_actions = env.act(obs)
        obs, rewards, done, info = env.step(all_actions)
        env.render()
    print(info)

env.close()