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
env = pommerman.make('OneVsOne-v0', agent_list)

for episode in tqdm(range(1000)):
    obs = env.reset()
    done = False
    while not done:
        all_actions = env.act(obs)
        obs, rewards, done, info = env.step(all_actions)
        print(obs[0])
        env.render()
    print(info)
print('1000 test ok')

env.close()

# import numpy as np
#
# f_path = 'dataset/hako_v2/228n5_5.npz'
# sub_data = np.load(f_path, allow_pickle=True)
# obs = sub_data['obs']
# actions = sub_data['actions']
# del sub_data
# print(obs[0])
# print(actions[0])
# print(obs.shape)
# print(actions.shape)
# print(obs[0].shape)
# print(actions[0].shape)


# if __name__ == '__main__':
#     arg_parser = arg_parser()
#     args, unknown_args = arg_parser.parse_known_args(sys.argv)
#     files_list = os.listdir(args.expert_path)
#     for f_name in files_list:
#         _thread.start_new_thread(_convert_dataset, f_name)
