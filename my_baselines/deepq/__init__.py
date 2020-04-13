from .policies import MlpPolicy, CnnPolicy, LnMlpPolicy, LnCnnPolicy
from .build_graph import build_act, build_train  # noqa
from .dqn import DQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa


def wrap_atari_dqn(env):
    """
    wrap the environment in atari wrappers for DQN

    :param env: (Gym Environment) the environment
    :return: (Gym Environment) the wrapped environment
    """
    from stable_baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
