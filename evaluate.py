import pommerman
from pommerman import agents
import sys
from my_common.cmd_utils import arg_parser
from my_baselines import PPO2
from my_common import featurize
from my_common import get_modify_act
from my_common import get_prev2obs
import time
from pommerman import constants
from my_agents import *
import utils


def make_env(env_id):
    agent_list = [
        SuicideAgent(),
        hit18Agent('0'),
        SuicideAgent(),
        hit18Agent('2')
    ]
    env = pommerman.make(env_id, agent_list)
    return env


def _evaluate():
    print('----------------------------------------------')
    print('|               E V A L U A T E              |')
    print('----------------------------------------------')
    env_id = 'PommeRadioCompetition-v4'
    env = utils.make_env(env_id)
    model_type = 'ppo'
    # model_path0 = 'models/pretrain_v3/pgn_v3_e79.zip'
    model_path0 = 'models/test/v14_2000k.zip'
    model_path1 = 'models/pretrain_v1/pgn_e118.zip'
    model_path2 = 'models/pretrain_v1/pgn_e118.zip'
    model_path3 = 'models/pretrain_v1/pgn_e118.zip'
    model_paths = [model_path0, model_path1, model_path2, model_path3]
    models = utils.get_load_models(model_type, model_paths)

    win = 0
    tie = 0
    loss = 0

    using_prune = False
    if using_prune:
        prune_agnets = [0, 1, 2, 3]  # 哪些智能体使用剪枝
        nokicks = [True, True, True, True]  # 调整是否使用kick
        print('prune_agents = ', prune_agnets)
        print('nokicks', nokicks)

    for episode in range(100):
        start = time.time()
        obs = env.reset()
        done = False
        prev2s = [(None, None)] * 4
        while not done:
            all_actions = env.act(obs)

            # Use model
            for i in range(len(models)):
                if models[i] is not None:
                    feature_obs = featurize(obs[i])
                    action, _states = models[i].predict(feature_obs)
                    if type(action) == list:
                        action = action[0]
                    all_actions[i] = action

            # Use prune
            if using_prune:
                for i in prune_agnets:
                    all_actions[i] = get_modify_act(obs[i], all_actions[i], prev2s[i], nokick=nokicks[i])
                    prev2s[i] = get_prev2obs(prev2s[i], obs[i])

            # 修正为适应通信的动作
            # if args.env == 'PommeRadioCompetition-v2':
            #     for i in range(len(all_actions)):
                    # all_actions[i] = [all_actions[i], 1, 1]

            obs, rewards, done, info = env.step(all_actions)

        if info['result'] == constants.Result.Tie:
            tie += 1
        elif info['winners'] == [0, 2]:
            win += 1
        else:
            loss += 1
        end = time.time()
        print("win / tie / loss")
        print(
            " %d  /  %d  /  %d  win rate: %f  use time: %f" % (win, tie, loss, (win / (episode + 1)), end - start))
    env.close()


if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _evaluate()
