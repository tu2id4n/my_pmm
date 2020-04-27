import sys
import os
import multiprocessing
import tensorflow as tf
from my_common.cmd_utils import arg_parser
from my_common.subproc_vec_env_v2 import SubprocVecEnv2
from my_baselines.ppo2_v2 import PPO2
from utils import make_envs, get_my_policy


def _learn():
    total_timesteps = int(args.num_timesteps)
    my_policy = get_my_policy(args.policy_type)
    # Mutiprocessing
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.allow_growth = True
    num_envs = args.num_env or multiprocessing.cpu_count()
    envs = [make_envs(args.env) for _ in range(num_envs)]
    env = SubprocVecEnv2(envs)

    if args.load_path:
        model = PPO2.load(load_path=args.load_path, using_pgn=args.using_pgn,
                          tensorboard_log=args.log_path, env=env)
    else:
        model = PPO2(my_policy, env=env, verbose=1, tensorboard_log=args.log_path, n_steps=128)

    model.learn(total_timesteps=total_timesteps, save_path=args.save_path, save_interval=args.save_interval)


if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _learn()
