import pommerman
from pommerman import agents
import sys
from my_common.cmd_utils import arg_parser
from my_baselines import PPO2
from my_common import featurize
from my_common import get_modify_act
from my_common import get_prev2obs
from my_agents import *

def make_env(env_id):
    agent_list = [
        hit18Agent('0'),
        agents.SimpleAgent(),
        hit18Agent('2'),
        agents.SimpleAgent()
    ]
    env = pommerman.make(env_id, agent_list)
    return env


def _play():
    env = make_env(args.env)
    models = load_models()
    print()
    print("env = ", args.env)
    print("**************** Start to play ****************")
    print('model0 is: ', args.model0)
    print('model1 is: ', args.model1)
    print('model2 is: ', args.model2)
    print('model3 is: ', args.model3)

    for episode in range(100):
        obs = env.reset()
        done = False
        prev2s = [(None, None)] * 4
        nokicks = [True] * 4
        while not done:
            all_actions = env.act(obs)

            # Use model
            for i in range(len(models)):
                if models[i] is not None:
                    feature_obs = featurize(obs)
                    action, _states = models[i].predict(feature_obs)
                    if type(action) == list:
                        action = action[0]
                    all_actions[i] = action
                    nokicks[i] = False

            # Use prune
            if args.using_prune:
                for i in range(len(all_actions)):
                    all_actions[i] = get_modify_act(obs[i], all_actions[i], prev2s[i], nokick=nokicks[i])
                    prev2s[i] = get_prev2obs(prev2s[i], obs[i])

            obs, rewards, done, info = env.step(all_actions)
            env.render()
        print(info)
    env.close()


def load_models():
    if args.model0_path:
        print()
        print("Load a model0 from", args.model0_path)
        model0 = PPO2.load(args.model0_path, using_pgn=args.using_pgn, tensorboard_log=args.log_path)
    else:
        print()
        print("No model0")
        model0 = None
    if args.model1_path:
        print()
        print("Load a model1 from", args.model1_path)
        model1 = PPO2.load(args.model1_path, using_pgn=args.using_pgn, tensorboard_log=args.log_path)
    else:
        print()
        print("No model1")
        model1 = None
    if args.model2_path:
        print()
        print("Load a model2 from", args.model2_path)
        model2 = PPO2.load(args.model2_path, using_pgn=args.using_pgn, tensorboard_log=args.log_path)
    else:
        print()
        print("No model2")
        model2 = None
    if args.model3_path:
        print()
        print("Load a model3 from", args.model3_path)
        model3 = PPO2.load(args.model3_path, using_pgn=args.using_pgn, tensorboard_log=args.log_path)
    else:
        print()
        print("No model3")
        model3 = None

    return model0, model1, model2, model3


if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _play()
