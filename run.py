import sys
import os
import multiprocessing
import tensorflow as tf
from my_common.cmd_utils import arg_parser
from my_common.subproc_vec_env import SubprocVecEnv
from utils import *


def _learn():
    total_timesteps = int(args.num_timesteps)
    my_policy = get_my_policy(args.policy_type)
    if args.model_type.lower() == 'ppo':
        # Mutiprocessing
        config = tf.ConfigProto()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config.gpu_options.allow_growth = True
        num_envs = args.num_env or multiprocessing.cpu_count()
        envs = [make_envs(args.env) for _ in range(num_envs)]
    else:
        envs = [make_envs(args.env)]

    env = SubprocVecEnv(envs)

    if args.load_path:
        model = get_load_model(args.model_type, args.load_path, args.log_path, env=env, using_pgn=args.using_pgn)
    else:
        model = get_init_model(env, args.model_type, my_policy, args.log_path)

    model.learn(total_timesteps=total_timesteps, save_path=args.save_path, save_interval=args.save_interval)
    env.close()


if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _learn()
