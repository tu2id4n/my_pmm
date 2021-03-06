import sys
import utils
from my_common import feature_utils
from my_common import cmd_utils
from my_baselines.deepq import DQN
import numpy as np
import my_agents
import pommerman
from pommerman import agents
from pommerman import constants
from tqdm import tqdm
def _play():
    print('----------------------------------------------')
    print('|                  P L A Y                   |')
    print('----------------------------------------------')
    env_id = 'OneVsOne-v8'
    print('env = ', env_id)
    agent_list = [
        my_agents.StopAgent(),
        # my_agents.StopAgent(),
        my_agents.SimpleNoBombAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # hit18Agent('1'),
        # hit18Agent('3')
    ]
    env = pommerman.make(env_id, agent_list)
    # env = utils.make_env(env_id)

    vb = True
    model_path = 'models/dqn/kl_135k.zip'
    model = DQN.load(load_path=model_path)

    win = 0
    tie = 0
    loss = 0
    for episode in tqdm(range(100)):
        obs = env.reset()
        done = False
        total_reward = 0
        action_abs = 65
        while not done:
            all_actions = env.act(obs)
            # 规格化输入
            feature_obs = feature_utils.featurize(obs[0])
            if action_abs == 65:
                goal_map = np.zeros((8, 8))
                goal_map[(1, 3)] = 1

            else:
                goal = feature_utils.extra_goal(action_abs, obs[0], rang=8)  # 加入目标
                goal_map = np.zeros((8, 8))
                goal_map[goal] = 1
            goal_map = goal_map.reshape(1, 8, 8)
            feature_obs = np.concatenate((feature_obs, goal_map))
            feature_obs = feature_obs.transpose((1, 2, 0))
            # 模型预测
            action_abs, _states = model.predict(feature_obs)
            # goal = feature_utils.extra_goal(action_abs, obs[0], rang=8)
            # print(obs[0])
            # action = feature_utils._djikstra_act(obs[0], action_abs, rang=8)
            # action = feature_utils._djikstra_act_8m8(obs_nf=obs[0], goal_abs=action_abs)

            if type(action_abs) == list:
                action_abs = action_abs[0]

            all_actions[0] = action_abs
            # print('act_abs', all_actions[0])
            obs, rewards, done, info = env.step(all_actions)
            # print('reward', rewards[0])
            # print()
            env.render()
            total_reward += rewards[0]
            # if not env._agents[0].is_alive:
            #     done = True
        # print(info)
        # print('total_reward', total_reward)
        print(info['result'])
        print()
        if info['result'] == constants.Result.Win:
            win += 1
        elif info['result'] == constants.Result.Tie:
            tie += 1
        elif info['result'] == constants.Result.Loss:
            loss += 1
    env.close()
    print('Win rate:', win/100)
    print('Tie rate:', tie / 100)
    print('Loss rate:', loss / 100)

if __name__ == '__main__':
    arg_parser = cmd_utils.arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _play()
