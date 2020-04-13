import pommerman
from pommerman import agents
from pommerman import constants
from my_agents import *
from tqdm import tqdm
agent_list = [
    # agents.DockerAgent('multiagentlearning/hakozakijunctions', port=1021),
    # agents.DockerAgent('tu2id4n/hit_pmm:fix2', port=1023),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    # agents.DockerAgent('multiagentlearning/hakozakijunctions', port=1023),
    # agents.DockerAgent('tu2id4n/hit_pmm:fix2', port=1025),
    agents.SimpleAgent()
    # agents.SimpleAgent()
]
env = pommerman.make('PommeRadioCompetition-v2', agent_list)

for episode in tqdm(range(1000)):
    obs = env.reset()
    done = False
    while not done:
        all_actions = env.act(obs)
        obs, rewards, done, info = env.step(all_actions)
        env.render()
    print(info)
print('1000 test ok')

env.close()