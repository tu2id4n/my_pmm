import numpy as np
import copy
import random
from my_common import feature_utils


class HindSightBuffer(object):
    def __init__(self, n_steps, gamma, lam):
        self.obs_nf = None
        self.masks = None
        self.actions = None
        self.values = None
        self.neglogpaces = None
        self.rewards = None
        self.obs_nf = None
        self.last_values = None
        self.dones = None
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam

    def add(self, obs, masks, actions, values, neglogpaces, rewards, obs_nf, last_values, dones):
        self.obs = copy.deepcopy(obs)
        self.masks = copy.deepcopy(masks)
        self.actions = copy.deepcopy(actions)
        self.values = copy.deepcopy(values)
        self.neglogpaces = copy.deepcopy(neglogpaces)
        self.rewards = copy.deepcopy(rewards)
        self.obs_nf = copy.deepcopy(obs_nf)
        self.last_values = copy.deepcopy(last_values)
        self.dones = copy.deepcopy(dones)

    '''
    HindSight
    1: 连续做一个动作获得越来越多reward
    2: 做一个动作看之后有没有达到这个位置
    '''

    def run(self):
        # 记录所有dones的位置
        start = 0
        frac = []
        for j in range(len(self.dones)):
            temp_frac = []
            for i in range(len(self.masks)):
                if self.masks[i][j]:
                    temp_frac.append((start, i))
                    start = i
            if self.dones[j]:
                temp_frac.append((start, len(self.masks)))
            frac.append(temp_frac)
        for i in range(len(frac)):
            for fr in frac[i]:
                st, ed = fr
                if ed > st:
                    # 在一个episode中随机取出一帧画面
                    rand = random.randint(st, ed - 1)
                    act_abs = self.actions[rand][i]
                    goal = feature_utils.extra_goal(act_abs, self.obs_nf[rand][i])
                    for j in range(rand, ed):
                        if self.obs_nf[j][i]['position'] == goal:
                            self.rewards[j][i] += 0.2
                            feature_utils.print_info('hindsight: 到达之前制定的goal, +0.2', vb=True)
        mb_advs = np.zeros_like(self.rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = self.last_values
            else:
                nextnonterminal = 1.0 - self.masks[step + 1]
                nextvalues = self.values[step + 1]
            # ∆ = r + 𝛄 * v' - v
            delta = self.rewards[step] + self.gamma * nextvalues * nextnonterminal - self.values[step]
            # adv = ∆ + 𝛄 * lam * adv-pre
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + self.values
        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs = map(self.swap_and_flatten, (
            self.obs, mb_returns, self.masks, self.actions, self.values, self.neglogpaces))
        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs

    def swap_and_flatten(self, arr):
        """
        swap and then flatten axes 0 and 1

        :param self: (np.ndarray)
        :return: (np.ndarray)
        """
        shape = arr.shape
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])