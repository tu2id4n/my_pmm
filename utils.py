import pommerman
from my_agents import *
from pommerman import agents
from my_policies import PGNPolicy, ResNetPolicy
from my_baselines import PPO2, DQN
from my_baselines.deepq.policies import CnnPolicy


def make_env(env_id):
    print('env = ', env_id)
    agent_list = [
        StopAgent(),
        # hit18Agent('1'),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # hit18Agent('3')
    ]
    env = pommerman.make(env_id, agent_list)
    return env


def make_envs(env_id):
    print('env = ', env_id)

    def _thunk():
        agent_list = [
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
        ]
        env = pommerman.make(env_id, agent_list)
        return env

    return _thunk


def get_my_policy(policy_type):
    if policy_type.lower() == 'resnet':
        return ResNetPolicy
    elif policy_type.lower() == 'pgn':
        return PGNPolicy

    print("policy_type = ", policy_type)
    print()


def get_init_model(env, model_type, my_policy, log_path):
    if model_type.lower() == 'ppo':
        model = PPO2(my_policy, env=env, verbose=1, tensorboard_log=log_path)
    elif model_type.lower() == 'dqn':
        model = DQN(CnnPolicy, env=env, verbose=1, tensorboard_log=log_path, buffer_size=2000, param_noise=True,
                    batch_size=64, train_freq=50, target_network_update_freq=500, gamma=0.99)
    return model


def get_load_model(model_type, load_path, log_path, env=None, using_pgn=False):
    if model_type.lower() == 'ppo':
        model = PPO2.load(load_path=load_path, using_pgn=using_pgn, tensorboard_log=log_path, env=env)
    elif model_type.lower() == 'dqn':
        model = DQN.load(load_path=load_path, env=env)
    return model


def get_load_models(model_type, model_paths, log_path, env=None, using_pgn=False):
    count = 0
    models = []
    for model_path in model_paths:
        if model_path:
            models.append(get_load_model(model_type, model_path, log_path, env=env, using_pgn=using_pgn))
            print('model_path:', model_path)
        else:
            models.append(None)
            print("No model", count)
        count += 1
    print('models:', models)
    return models
