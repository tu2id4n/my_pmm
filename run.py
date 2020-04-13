import pommerman
from pommerman import agents
import sys
import os
import multiprocessing
import tensorflow as tf
from my_common.cmd_utils import arg_parser
from my_common.subproc_vec_env import SubprocVecEnv
from my_policies import PGNPolicy, ResNetPolicy
from my_baselines import PPO2, DQN
from stable_baselines import SAC
from my_baselines.deepq.policies import CnnPolicy

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
    if args.policy_type.lower() == 'resnet':
        return ResNetPolicy
    elif args.policy_type.lower() == 'pgn':
        return PGNPolicy

def get_model(model_type, env):
    if model_type.lower() == 'ppo':
        if args.load_path:
            model = PPO2.load(args.load_path, env=env, using_pgn=args.using_pgn, tensorboard_log=args.log_path)
        else:
            model = PPO2(get_my_policy(), env=env, verbose=1, tensorboard_log=args.log_path)
    elif model_type.lower() == 'dqn':
        model = DQN(CnnPolicy, env=env, verbose=1, tensorboard_log=args.log_path)
    return model


def _learn():
    # Mutiprocessing
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.allow_growth = True
    num_envs = args.num_env or multiprocessing.cpu_count()
    # envs = [make_envs(args.env) for _ in range(num_envs)]
    envs = [make_envs(args.env)]
    env = SubprocVecEnv(envs)

    total_timesteps = int(args.num_timesteps)
    model = get_model(args.model_type, env)

    print("policy_type = ", args.policy_type)
    print('env = ', args.env)

    model.learn(total_timesteps=total_timesteps, save_path=args.save_path, save_interval=args.save_interval)
    env.close()



if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _learn()
