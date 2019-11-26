import pommerman
from pommerman import agents
from pommerman import constants
from my_agents import *

agent_list = [
    # agents.DockerAgent('tu2id4n/hit-pmm:fix', port=12349),
    agents.DockerAgent('hit-pmm/mhp_v1', port=12345),
    agents.SimpleAgent(),
    SuicideAgent(),
    agents.SimpleAgent()
]
env = pommerman.make('PommeRadioCompetition-v2', agent_list)

for episode in range(100):
    obs = env.reset()
    done = False
    while not done:
        all_actions = env.act(obs)
        obs, rewards, done, info = env.step(all_actions)
        # env.render()
    print(info)

env.close()