import pommerman
from pommerman import agents
import sys
import os
import multiprocessing
import tensorflow as tf
from my_common.cmd_utils import arg_parser
from my_common.subproc_vec_env import SubprocVecEnv
from my_policies import PGNPolicy, ResNetPolicy
from my_baselines import PPO2


def make_envs(env_id):
    def _thunk():
        agent_list = [
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
        ]
        env = pommerman.make(env_id, agent_list)
        return env

    return _thunk


def get_my_policy():
    if args.policy_type == 'resnet':
        return ResNetPolicy
    elif args.policy_type == 'pgn':
        return PGNPolicy


def _learn():
    # Mutiprocessing
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.allow_growth = True
    num_envs = args.num_env or multiprocessing.cpu_count()
    envs = [make_envs(args.env) for _ in range(num_envs)]
    env = SubprocVecEnv(envs)

    total_timesteps = int(args.num_timesteps)
    if args.load_path:
        print()
        print("Load a model from", args.load_path)
        print()
        model = PPO2.load(args.load_path, env=env, using_pgn=args.using_pgn, tensorboard_log=args.log_path,
                          policy=get_my_policy())
    else:
        model = PPO2(get_my_policy(), env=env, verbose=1, tensorboard_log=args.log_path)

    print()
    print("**************** Start to learn ****************")
    print("num timesteps = ", args.num_timesteps)
    print("policy_type = ", args.policy_type)
    print("env = ", args.env)
    print("num_envs = ", num_envs)
    print("save_interval = ", args.save_interval)
    print()

    model.learn(total_timesteps=total_timesteps, save_path=args.save_path, save_interval=args.save_interval)

    env.close()



if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _learn()
