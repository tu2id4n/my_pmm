import sys
import os
import multiprocessing
import tensorflow as tf
from my_common.cmd_utils import arg_parser
from my_common.subproc_vec_env_8m8 import SubprocVecEnv
from utils import *
from my_baselines.deepq import DQN, CnnPolicy


def _learn():
    total_timesteps = int(args.num_timesteps)
    # my_policy = get_my_policy(args.policy_type)
    # # Mutiprocessing
    # config = tf.ConfigProto()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # config.gpu_options.allow_growth = True
    # num_envs = args.num_env or multiprocessing.cpu_count()
    # envs = [make_envs(args.env) for _ in range(num_envs)]
    env = SubprocVecEnv([make_envs('OneVsOne-v8')])

    model = DQN(env=env, policy=CnnPolicy, tensorboard_log=args.log_path, buffer_size=50000,
                param_noise=False, verbose=1,
                train_freq=1, target_network_update_freq=500, gamma=0.99,
                exploration_fraction=0.1, exploration_final_eps=0.02,
                learning_starts=500, temp_size=5, k=1)

    model.learn(total_timesteps=total_timesteps, save_path=args.save_path, save_interval=args.save_interval)
    env.close()


if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _learn()
