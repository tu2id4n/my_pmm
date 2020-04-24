from pommerman import constants
import numpy as np
from my_common import feature_utils
import copy
import queue


def print_info(name, info, Visualize=True):
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


# 即时奖励, 根据目的设计奖励
def get_rewards_v3_6(agents, step_count, max_steps, whole_obs_pre, whole_obs, act_abs_pre):
    def any_lst_equal(lst, values):
        """Checks if list are equal"""
        return any([lst == v for v in values])

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]

    obs_pre = copy.deepcopy(whole_obs_pre[0])
    obs_now = copy.deepcopy(whole_obs[0])

    position_pre = obs_pre['position']
    position_now = obs_now['position']

    bomb_life_now = feature_utils.get_bomb_life(obs_now)
    bomb_life_pre = feature_utils.get_bomb_life(obs_pre)
    my_bomb_life_now = feature_utils.get_my_bomb_life(bomb_life_now, position_now)

    extrabomb = constants.Item.ExtraBomb.value
    kick = constants.Item.Kick.value
    incrrange = constants.Item.IncrRange.value
    wood = constants.Item.Wood.value
    agent1 = constants.Item.Agent1.value
    agent3 = constants.Item.Agent3.value
    e11_now = feature_utils.extra_position(11, obs_now['board'])
    e13_now = feature_utils.extra_position(13, obs_now['board'])

    reward = 0
    # # 敌人被炸死
    # if e11 is not None and 0 < bomb_life[e11] < 4:
    #     reward += 0.5
    #     print_info('e11被炸死', '+0.5')
    # if e13 is not None and 0 < bomb_life[e13] < 4:
    #     reward += 0.5
    #     print_info('e13被炸死', '+0.5')

    # 自己被炸死
    if 0 < bomb_life_now[position_now] < 4:
        reward -= 0.5
        print_info('自己被炸死', '-0.5')

    act_pre = feature_utils._djikstra_act(obs_pre, act_abs_pre)  # 这里只用来判断
    goal_pre = feature_utils.extra_goal(act_abs_pre, obs_pre)
    # 如果是放bomb
    if act_pre == 5:
        # 没有ammo放bomb
        if obs_pre['ammo'] == 0:
            reward -= 0.1
            print_info('没有ammo放炸弹', '-0.1')
        # 如果有ammo
        else:
            # 放的bomb可以波及到wood/enemy
            for r in range(11):
                for c in range(11):
                    if my_bomb_life_now[(r, c)] > 0:
                        if obs_pre['board'][(r, c)] in [wood]:
                            reward += 0.2
                            print_info('炸弹波及到wood', '+0.2')
                        if obs_pre['board'][(r, c)] in [agent1, agent3]:
                            reward += 0.3
                            print_info('炸弹波及到敌人', '+0.3')
    # 没有动
    elif act_pre == 0:
        if obs_pre['position'] != goal_pre:
            reward -= 0.01
            print_info('无效移动', '-0.01')
    # 如果是移动
    else:
        # 有效的移动
        # reward += 0.001
        # print_info('有效的移动', '+0.001')
        # 被炸弹波及但是在向安全的位置移动
        if bomb_life_pre[position_pre] > 0 and bomb_life_pre[goal_pre] == 0:
            reward += 0.05
            print_info('被炸弹波及向着安全的位置移动', '+0.05')
        # 向着items移动
        if obs_pre['board'][goal_pre] in [extrabomb, kick, incrrange]:
            reward += 0.01
            print_info('向items移动', '+0.01')
            # 吃到items
            if obs_pre['board'][position_now] in [extrabomb, kick, incrrange]:
                reward += 0.3
                print_info('向着item移动并吃到items', '+0.2')
        # 吃到items
        elif obs_pre['board'][position_now] in [extrabomb, kick, incrrange]:
            reward += 0.05
            print_info('路过吃到items', '+0.05')

    # We are playing a team game.
    if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
        # Team [0, 2] wins.
        print_info('Team [0, 2] wins and agent0 alive.', reward + 1)
        return [reward + 1, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[1, 3]]):
        # Team [1, 3] wins and no enemy dead.
        print_info('Team [1, 3] wins and no enemy dead.', reward - 1)
        return [reward - 1, 1, -1, 1]
    elif any_lst_equal(alive_agents, [[1], [3]]):
        # Team [1, 3] wins and one enemy dead.
        print_info('Team [1, 3] wins and one enemy dead.', reward + 0.5)
        return [reward + 0.5, 1, -1, 1]
    elif step_count >= max_steps and any_lst_equal(alive_agents, [[0, 1], [0, 1, 2], [0, 3], [0, 2, 3]]):
        # tie and one enemy dead.
        print_info('tie and one enemy dead.', reward + 0.5)
        return [reward + 0.5, 1, -1, 1]
    elif step_count >= max_steps:
        # Game is over by max_steps. All agents tie.
        print_info('Game is over by max_steps. All agents tie.', reward - 1)
        return [reward - 1] * 4
    elif len(alive_agents) == 0:
        # Everyone's dead. All agents tie.
        print_info('Everyone is dead. All agents tie.', reward + 0.5)
        return [reward + 0.5] * 4
    else:
        # No team has yet won or lost.
        return [reward] * 4


# 即时奖励, 根据目的设计奖励, 升级版本, 使用_djikstra_v2探路
def get_rewards_v3_7(agents, step_count, max_steps, whole_obs_pre, whole_obs, act_abs_pre):
    def any_lst_equal(lst, values):
        """Checks if list are equal"""
        return any([lst == v for v in values])

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]

    obs_pre = copy.deepcopy(whole_obs_pre[0])
    obs_now = copy.deepcopy(whole_obs[0])

    position_pre = obs_pre['position']
    position_now = obs_now['position']

    bomb_life_now = feature_utils.get_bomb_life(obs_now)
    bomb_life_pre = feature_utils.get_bomb_life(obs_pre)
    my_bomb_life_now = feature_utils.get_my_bomb_life(bomb_life_now, position_now)

    extrabomb = constants.Item.ExtraBomb.value
    kick = constants.Item.Kick.value
    incrrange = constants.Item.IncrRange.value
    bomb = constants.Item.Bomb.value
    wood = constants.Item.Wood.value
    agent1 = constants.Item.Agent1.value
    agent3 = constants.Item.Agent3.value
    agent2 = constants.Item.Agent2.value
    e11_pre = feature_utils.extra_position(11, obs_pre['board'])
    e13_pre = feature_utils.extra_position(13, obs_pre['board'])
    e11_now = feature_utils.extra_position(11, obs_now['board'])
    e13_now = feature_utils.extra_position(13, obs_now['board'])

    reward = 0
    # # 敌人被炸死
    # if e11 is not None and 0 < bomb_life[e11] < 4:
    #     reward += 0.5
    #     print_info('e11被炸死', '+0.5')
    # if e13 is not None and 0 < bomb_life[e13] < 4:
    #     reward += 0.5
    #     print_info('e13被炸死', '+0.5')

    # 敌人从视野中消失:
    # if e11_now is None and e11_pre is not None:
    #     reward -= 0.02
    #     print_info('敌人e11消失', '-0.01')
    # if e13_now is None and e13_pre is not None:
    #     reward -= 0.02
    #     print_info('敌人e13消失', '-0.01')
    # if e11_pre is None and e11_now is not None:
    #     reward += 0.01
    #     print_info('敌人e11出现', '+0.01')
    # if e13_pre is None and e13_now is not None:
    #     reward += 0.01
    #     print_info('敌人e13出现', '+0.01')

    # 自己被炸死
    if 0 < bomb_life_now[position_now] < 4:
        reward -= 1
        print_info('自己被炸死', '-1')

    act_pre = feature_utils._djikstra_act(obs_pre, act_abs_pre)  # 这里只用来判断
    goal_pre = feature_utils.extra_goal(act_abs_pre, obs_pre)
    # 如果是放bomb
    if act_pre == 5:
        # 没有ammo放bomb
        if obs_pre['ammo'] == 0:
            reward -= 0.1
            print_info('没有ammo放炸弹', '-0.1')
        # 如果有ammo
        else:
            nothing = True
            # 放的bomb可以波及到wood/enemy
            for r in range(11):
                for c in range(11):
                    if my_bomb_life_now[(r, c)] > 0:
                        if obs_pre['board'][(r, c)] in [wood]:
                            reward += 0.2
                            nothing = False
                            print_info('bomb波及到wood', '+0.2')
                        if obs_pre['board'][(r, c)] in [agent1, agent3]:
                            reward += 0.3
                            nothing = False
                            print_info('bomb波及到enemy', '+0.3')
                        if obs_pre['board'][(r, c)] in [incrrange, extrabomb, kick]:
                            reward -= 0.1
                            print_info('bomb波及powerup', '-0.1')
                        if obs_pre['board'][(r, c)] in [agent2]:
                            reward -= 0.1
                            print_info('bomb波及teammates', '-0.1')
            if nothing:
                reward -= 0.2
                print_info('空放bomb', '-0.2')
    # 没有动
    elif act_pre == 0:
        if obs_pre['position'] != goal_pre:
            reward -= 0.1
            print_info('无效移动', '-0.1')
    # 如果是移动
    else:
        # r_pre, c_pre = position_pre
        # r_now, c_now = position_now
        # r_to = r_now - r_pre
        # c_to = c_now - c_pre
        # if (r_to, c_to) == (-1, 0): act_pre = 1
        # if (r_to, c_to) == (1, 0): act_pre = 2
        # if (r_to, c_to) == (0, -1): act_pre = 3
        # if (r_to, c_to) == (0, 1): act_pre = 4
        # 有效的移动
        # reward += 0.001
        # print_info('有效的移动', '+0.001')
        # 踢炸弹获得奖励
        if obs_pre['can_kick']:
            if obs_pre['board'][goal_pre] == bomb:
                reward += 0.05
                print_info('想去踢bomb', '+0.05')
            if obs_pre['board'][position_now] == bomb:
                reward += 0.05
                print_info('踢到炸弹', '+0.05')
        # 从安全位置进入到被炸弹波及之中
        if bomb_life_pre[position_pre] == 0 and bomb_life_now[position_now] > 0:
            reward -= 0.12
            print_info('从安全位置进入到bomb波及范围', '-0.12')
        # 被炸弹波及但是在向安全的位置移动
        if bomb_life_pre[position_pre] > 0 and bomb_life_pre[goal_pre] == 0:
            reward += 0.05
            print_info('被bomb波及向着安全的位置移动', '+0.05')
        # 向着items移动
        if obs_pre['board'][goal_pre] in [extrabomb, kick, incrrange]:
            reward += 0.05
            print_info('向items移动', '+0.05')
            # 吃到items
            if obs_pre['board'][position_now] in [extrabomb, kick, incrrange]:
                reward += 0.05
                print_info('向着item移动并吃到items', '+0.05')
        # 吃到items
        elif obs_pre['board'][position_now] in [extrabomb, kick, incrrange]:
            reward += 0.05
            print_info('路过吃到items', '+0.05')

    # We are playing a team game.
    if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
        # Team [0, 2] wins.
        print_info('Team [0, 2] wins and agent0 alive.', reward + 1)
        return [reward + 1, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[1, 3]]):
        # Team [1, 3] wins and no enemy dead.
        print_info('Team [1, 3] wins and no enemy dead.', reward - 1)
        return [reward - 1, 1, -1, 1]
    elif any_lst_equal(alive_agents, [[1], [3]]):
        # Team [1, 3] wins and one enemy dead.
        print_info('Team [1, 3] wins and one enemy dead.', reward - 0.6)
        return [reward - 0.6, 1, -1, 1]
    elif step_count >= max_steps and any_lst_equal(alive_agents, [[0, 1], [0, 1, 2], [0, 3], [0, 2, 3]]):
        # tie and one enemy dead.
        print_info('tie and one enemy dead.', reward - 0.6)
        return [reward - 0.6, 1, -1, 1]
    elif step_count >= max_steps:
        # Game is over by max_steps. All agents tie.
        print_info('Game is over by max_steps. All agents tie.', reward - 1)
        return [reward - 1] * 4
    elif len(alive_agents) == 0:
        # Everyone's dead. All agents tie.
        print_info('Everyone is dead. All agents tie.', reward)
        return [reward] * 4
    else:
        # No team has yet won or lost.
        return [reward] * 4
