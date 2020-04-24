import multiprocessing
from collections import OrderedDict

from gym import spaces
from pommerman import *

from stable_baselines.common.vec_env import VecEnv, CloudpickleWrapper
from stable_baselines.common.tile_images import tile_images

from my_common.feature_utils import *


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    # TODO:记得设置训练智能体的 index
    train_idx = 0  # 设置训练的 agent 的 index
    teammates = [train_idx, (train_idx + 2) % 4]
    teammates.sort()
    enemies = [(train_idx + 1) % 4, (train_idx + 3) % 4]
    enemies.sort()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                whole_obs = env.get_observations()
                all_actions = env.act(whole_obs)  # 得到所有智能体的 actions

                # 如果其他智能体动作不是元组（只有单一动作），改成list
                for i in range(4):
                    if not isinstance(all_actions[i], list):
                        all_actions[i] = [all_actions[i], 0, 0]

                # data = _djikstra_act(whole_obs[train_idx], data)  # v4 则取消_djikstra
                # data = all_actions[i] = get_modify_act(whole_obs[train_idx], data)
                data = [data, 0, 0]
                all_actions[train_idx] = data  # 当前训练的 agent 的动作也加进来
                whole_obs, whole_rew, done, info = env.step(all_actions)  # 得到所有 agent 的四元组
                rew = whole_rew[train_idx]  # 得到训练智能体的当前步的 reward
                win_rate = 0  # 输出胜率
                tie_rate = 0  # 输出平局
                loss_rate = 0  # 输出输率
                first_dead_rate = 0
                # 判断智能体是否死亡, 死亡则结束，并将奖励设置为-1
                if not done and not env._agents[train_idx].is_alive:
                    done = True
                    # 如果先死则增加死亡几率
                    first_dead_rate = 1
                    # rew = rew - 1

                if done:  # 如果结束, 重新开一把
                    info['terminal_observation'] = whole_obs  # 保存终结的 observation，否则 reset 后将丢失
                    # if info['winners'] == enemies:
                    #     info = constants.Result.Loss
                    if info['result'] == constants.Result.Win:
                        win_rate = 1
                    elif info['result'] == constants.Result.Loss:
                        loss_rate = 1
                    elif info['result'] == constants.Result.Tie:
                        tie_rate += 1
                    whole_obs = env.reset()  # 重新开一把

                obs = featurize(whole_obs[train_idx])
                # remote.send((obs, rew, done, win_rate))
                remote.send((obs, rew, done, win_rate, tie_rate, loss_rate, first_dead_rate, whole_obs[train_idx]))

            elif cmd == 'reset':
                whole_obs = env.reset()
                obs = featurize(whole_obs[train_idx])

                # remote.send(obs)
                remote.send((obs, whole_obs[train_idx]))

            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                """增加前三行，注释最后一行。自定义 observation 和 action 的 space"""
                observation_space = get_observertion_space()
                action_space = get_action_space()
                remote.send((observation_space, action_space))
                # remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            """注释前两行，增加第三行"""
            # forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            # start_method = 'forkserver' if forkserver_available else 'spawn'
            start_method = 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, win_rate, tie_rate, loss_rate, first_dead_rate, obs_nf = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), np.stack(win_rate), np.stack(
            tie_rate), np.stack(loss_rate), np.stack(first_dead_rate), obs_nf

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, obs_nf = zip(*results)
        return _flatten_obs(obs, self.observation_space), obs_nf

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human', *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
