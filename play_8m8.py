import sys
import utils
from my_common import feature_utils
from my_common import cmd_utils
from my_baselines.deepq import DQN
import numpy as np

def _play():
    print('----------------------------------------------')
    print('|                  P L A Y                   |')
    print('----------------------------------------------')
    env_id = 'OneVsOne-v0'
    env = utils.make_env(env_id)

    vb = True
    model_path = 'models/_200k.zip'
    model = DQN.load(load_path=model_path)

    for episode in range(100):
        obs = env.reset()
        done = False
        total_reward = 0
        action_abs = 65
        while not done:
            all_actions = env.act(obs)
            feature_obs = feature_utils.featurize(obs[0])
            if action_abs == 65:
                feature_obs = np.concatenate((feature_obs, np.zeros((1, 8, 8))))
            else:
                goal = feature_utils.extra_goal_8m8(action_abs, obs[0])  # 加入目标
                goal_map = np.zeros((8, 8))
                goal_map[(goal)] = 1
                goal_map = goal_map.reshape(1, 8, 8)
                feature_obs = np.concatenate((feature_obs, goal_map))
            feature_obs = np.stack(feature_obs, axis=2)
            action_abs, _states = model.predict(feature_obs)
            action = feature_utils._djikstra_act_8m8(obs_nf=obs[0], goal_abs=action_abs)
            utils.print_info('action_obs', action_abs, vb)
            utils.print_info('action', action, vb)
            if type(action) == list:
                action = action[0]

            all_actions[0] = action

            obs, rewards, done, info = env.step(all_actions)
            env.render()
            total_reward += rewards[0]
            if not env._agents[0].is_alive:
                done = True
        print(info)
        print('total_reward', total_reward)
    env.close()


if __name__ == '__main__':
    arg_parser = cmd_utils.arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _play()
