from pommerman import constants
import numpy as np
from my_common import feature_utils
import copy
import queue


def print_info(name, info, Visualize=False):
    if Visualize:
        print(name, info)


# skynet 课程设计
def get_rewards_v3_1(agents, step_count, max_steps, obs_pre, obs, position_trav, action_pre):
    def any_lst_equal(lst, values):
        '''Checks if list are equal'''
        return any([lst == v for v in values])

    reward = 0

    position_new = obs[0]['position']
    if position_new not in position_trav:
        reward += 0.001
        print_info('position')

    if obs_pre[0]['can_kick'] is False and obs[0]['can_kick'] is True:
        reward += 0.01
        print_info('can_kick')

    if obs_pre[0]['board'][position_new] == constants.Item.ExtraBomb.value:
        reward += 0.01
        print_info('extrabomb')

    if obs_pre[0]['blast_strength'] < obs[0]['blast_strength']:
        reward += 0.01
        print_info('blast_strength')

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]

    # We are playing a team game.
    if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
        # Team [0, 2] wins.
        print_info('Team [0, 2] wins and agent0 alive.')
        return [reward + 1, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[1, 3]]):
        # Team [1, 3] wins and no enemy dead.
        print_info('Team [1, 3] wins and no enemy dead.')
        return [reward - 1, 1, -1, 1]
    elif any_lst_equal(alive_agents, [[1], [3]]):
        # Team [1, 3] wins and one enemy dead.
        print_info('Team [1, 3] wins and one enemy dead.')
        return [reward + 0.5, 1, -1, 1]
    elif step_count >= max_steps and any_lst_equal(alive_agents, [[0, 1], [0, 1, 2], [0, 3], [0, 2, 3]]):
        # tie and one enemy dead.
        print_info('tie and one enemy dead.')
        return [reward + 0.5, 1, -1, 1]
    elif step_count >= max_steps:
        # Game is over by max_steps. All agents tie.
        print_info('Game is over by max_steps. All agents tie.')
        return [reward - 1] * 4
    elif len(alive_agents) == 0:
        # Everyone's dead. All agents tie.
        print_info('Everyone is dead. All agents tie.')
        return [reward + 1] * 4
    else:
        # No team has yet won or lost.
        print_info('No team has yet won or lost.')
        return [reward] * 4


# combat 课程设计
def get_rewards_v3_2(agents, step_count, max_steps, obs_pre, obs, position_trav, action_pre):
    def any_lst_equal(lst, values):
        '''Checks if list are equal'''
        return any([lst == v for v in values])

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]
    reward = 0
    position_new = obs[0]['position']

    if obs_pre[0]['can_kick'] is False and obs[0]['can_kick'] is True:
        reward += 0.1
        print_info('can_kick')

    if obs_pre[0]['board'][position_new] == constants.Item.ExtraBomb.value:
        reward += 0.05
        print_info('extrabomb')

    if obs_pre[0]['blast_strength'] < obs[0]['blast_strength']:
        reward += 0.05
        print_info('blast_strength')

    # We are playing a team game.
    if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
        # Team [0, 2] wins.
        return [1 + reward, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
        # Team [1, 3] wins.
        return [-1 + reward, 1, -1, 1]
    elif step_count >= max_steps:
        # Game is over by max_steps. All agents tie.
        return [-1 + reward] * 4
    elif len(alive_agents) == 0:
        # Everyone's dead. All agents tie.
        return [-1 + reward] * 4
    else:
        # No team has yet won or lost.
        return [0 + reward] * 4


# 吃powerup的课程设计
def get_rewards_v3_3(agents, step_count, max_steps, obs_pre, obs, position_trav, action_pre):
    def any_lst_equal(lst, values):
        """Checks if list are equal"""
        return any([lst == v for v in values])

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]
    reward = 0
    position_new = obs[0]['position']

    if obs_pre[0]['can_kick'] is False and obs[0]['can_kick'] is True:
        reward += 0.5
        print_info('can_kick')

    if obs_pre[0]['board'][position_new] == constants.Item.ExtraBomb.value:
        reward += 0.1
        print_info('extrabomb')

    if obs_pre[0]['blast_strength'] < obs[0]['blast_strength']:
        reward += 0.1
        print_info('blast_strength')

    return [reward] * 4


# 放炸弹课程设计
def get_rewards_v3_4(agents, step_count, max_steps, obs_pre, obs, position_trav, action_pre):
    def any_lst_equal(lst, values):
        """Checks if list are equal"""
        return any([lst == v for v in values])

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]
    reward = 0
    position_new = obs[0]['position']

    if obs_pre[0]['can_kick'] is False and obs[0]['can_kick'] is True:
        reward += 0.05
        print_info('can_kick')

    if obs_pre[0]['board'][position_new] == constants.Item.ExtraBomb.value:
        reward += 0.01
        print_info('extrabomb')

    if obs_pre[0]['blast_strength'] < obs[0]['blast_strength']:
        reward += 0.01
        print_info('blast_strength')

    position_pre = obs_pre[0]['position']
    ammo_pre = obs_pre[0]['ammo']
    x_pre, y_pre = position_pre
    if action_pre == 5:
        for act_obs in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = act_obs
            if obs_pre[0]['board'][(x_pre + r, y_pre + c)] in [constants.Item.Wood.value, constants.Item.Agent1.value,
                                                               constants.Item.Agent3.value]:
                reward += 0.1
                print_info('bomb')
        if ammo_pre == 0:
            reward -= 0.1
            print_info('no ammo but bomb')
    if ammo_pre > 0 and action_pre != 5:
        reward -= 0.02
        print_info('have ammo but dont bomb')

    board = np.array(obs[0]['board'])
    if (board == 11).any() or (board == 13).any():
        reward += 0.01

    # We are playing a team game.
    if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
        # Team [0, 2] wins.
        return [1 + reward, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
        # Team [1, 3] wins.
        return [-1 + reward, 1, -1, 1]
    elif step_count >= max_steps:
        # Game is over by max_steps. All agents tie.
        return [-1 + reward] * 4
    elif len(alive_agents) == 0:
        # Everyone's dead. All agents tie.
        return [-1 + reward] * 4
    else:
        # No team has yet won or lost.
        return [0 + reward] * 4


# 即时奖励, 放炸弹课程设计
def get_rewards_v3_5(agents, step_count, max_steps, obs_pre, obs, position_trav, action_pre):
    def any_lst_equal(lst, values):
        """Checks if list are equal"""
        return any([lst == v for v in values])

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]
    reward = 0
    position_new = obs[0]['position']

    if obs_pre[0]['can_kick'] is False and obs[0]['can_kick'] is True:
        reward += 0.1
        print_info('can_kick')

    if obs_pre[0]['board'][position_new] == constants.Item.ExtraBomb.value:
        reward += 0.1
        print_info('extrabomb')

    if obs_pre[0]['blast_strength'] < obs[0]['blast_strength']:
        reward += 0.1
        print_info('blast_strength')

    position_pre = obs_pre[0]['position']
    ammo_pre = obs_pre[0]['ammo']
    x_pre, y_pre = position_pre
    if action_pre == 5:
        for act_obs in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = act_obs
            if obs_pre[0]['board'][(x_pre + r, y_pre + c)] in [constants.Item.Wood.value, constants.Item.Agent1.value,
                                                               constants.Item.Agent3.value]:
                reward += 0.2
                print_info('bomb')
        if ammo_pre == 0:
            reward -= 0.02
            print_info('no ammo but bomb')
    if ammo_pre > 0 and action_pre != 5:
        reward -= 0.02
        print_info('have ammo but dont bomb')

    board = np.array(obs[0]['board'])
    if (board == 11).any() or (board == 13).any():
        reward += 0.001

    return [reward] * 4


# 即时奖励, 根据动作设计奖励
def get_rewards_v3_6(agents, step_count, max_steps, whole_obs_pre, whole_obs, act_abs_pre):
    def any_lst_equal(lst, values):
        """Checks if list are equal"""
        return any([lst == v for v in values])

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]
    obs_pre = copy.deepcopy(whole_obs_pre[0])
    obs = copy.deepcopy(whole_obs[0])
    # position_pre = obs_pre['position']
    position_new = obs['position']
    extrabomb = constants.Item.ExtraBomb.value
    kick = constants.Item.Kick.value
    incrrange = constants.Item.IncrRange.value
    wood = constants.Item.Wood.value
    agent1 = constants.Item.Agent1.value
    agent3 = constants.Item.Agent3.value
    act_pre = feature_utils._djikstra_act(obs_pre, act_abs_pre)
    bomb_life = feature_utils.get_bomb_life(obs)
    my_bomb_life = feature_utils.get_my_bomb_life(bomb_life, position_new)
    e11 = feature_utils.extra_position(11, obs['board'])
    e13 = feature_utils.extra_position(13, obs['board'])
    # print_info('e11位置', e11)
    # print_info('e13位置', e13)
    reward = 0
    # # 敌人被炸死
    # if e11 is not None and 0 < bomb_life[e11] < 4:
    #     reward += 0.5
    #     print_info('e11被炸死', '+0.5')
    # if e13 is not None and 0 < bomb_life[e13] < 4:
    #     reward += 0.5
    #     print_info('e13被炸死', '+0.5')
    # 自己被炸死
    if 0 < bomb_life[position_new] < 4:
        reward -= 1
        print_info('自己被炸死', '-1')

    if act_pre == 5:
        # 没有ammo放bomb
        if obs_pre['ammo'] == 0:
            reward -= 0.1
            print_info('没有ammo放炸弹', '-0.1')
        # 放的bomb可以波及到wood/enemy
        for r in range(11):
            for c in range(11):
                if my_bomb_life[(r, c)] > 0:
                    if obs_pre['board'][(r, c)] in [wood, agent1, agent3]:
                        reward += 0.2
                        print_info('炸弹波及到目标', '+0.2')
    # 向着items移动
    elif act_pre != 0:
        goal_pre = feature_utils.extra_goal(act_abs_pre)
        if obs_pre['board'][goal_pre] in [extrabomb, kick, incrrange]:
            reward += 0.01
            print_info('向items移动', '+0.01')
            # 吃到items
            if obs_pre['board'][position_new] in [extrabomb, kick, incrrange]:
                reward += 0.2
                print_info('向着item移动并吃到items', '+0.2')
    # 吃到items
    # if obs_pre['board'][position_new] in [extrabomb, kick, incrrange]:
    #     reward += 0.1
    #     print_info('向着item移动并吃到items', '+0.1')

    # We are playing a team game.
    if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
        # Team [0, 2] wins.
        return [1 + reward, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
        # Team [1, 3] wins.
        return [-1 + reward, 1, -1, 1]
    elif step_count >= max_steps:
        # Game is over by max_steps. All agents tie.
        return [-1 + reward] * 4
    elif len(alive_agents) == 0:
        # Everyone's dead. All agents tie.
        return [-1 + reward] * 4
    else:
        # No team has yet won or lost.
        return [0 + reward] * 4
