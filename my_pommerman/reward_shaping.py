from pommerman import constants
import numpy as np


def print_info(info, Visualize=False):
    if Visualize:
        print('reward change:', info)


def get_rewards_v3(agents, step_count, max_steps, obs_pre, obs, position_trav, action_pre):
    def any_lst_equal(lst, values):
        '''Checks if list are equal'''
        return any([lst == v for v in values])

    reward = 0

    position_new = obs[0]['position']
    if position_new not in position_trav:
        reward += 0.001
        print_info('position')

    if obs_pre[0]['can_kick'] is False and obs[0]['can_kick'] is True:
        reward += 0.02
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
                reward += 0.5
                print_info('bomb')
        if ammo_pre == 0:
            reward -= 0.2
            print_info('no ammo but bomb')
    if ammo_pre > 0 and action_pre != 5:
        reward -= 0.0001
        print_info('have ammo but dont bomb')

    board = np.array(obs[0]['board'])
    if (board == 11).any() or (board == 13).any():
        reward += 0.0001

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]

    # We are playing a team game.
    if any_lst_equal(alive_agents, [[0, 2], [0]]):
        # Team [0, 2] wins and agent0 alive.
        print_info('Team [0, 2] wins and agent0 alive.')
        return [reward + 1, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[2]]):
        # Team [0, 2] wins but agent0 is dead.
        print_info('Team [0, 2] wins but agent0 is dead.')
        return [reward + 0.5, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[1, 3]]):
        # Team [1, 3] wins and no enemy dead.
        print_info('Team [1, 3] wins and no enemy dead.')
        return [reward - 1, 1, -1, 1]
    elif any_lst_equal(alive_agents, [[1], [3]]):
        # Team [1, 3] wins and one enemy dead.
        print_info('Team [1, 3] wins and one enemy dead.')
        return [reward - 0.5, 1, -1, 1]
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
        return [reward] * 4
    else:
        # No team has yet won or lost.
        print_info('No team has yet won or lost.')
        return [reward] * 4

def get_rewards_v3_1(agents, step_count, max_steps, obs_pre, obs, position_trav, action_pre):
    def any_lst_equal(lst, values):
        '''Checks if list are equal'''
        return any([lst == v for v in values])

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]
    reward = 0
    position_new = obs[0]['position']

    if obs_pre[0]['can_kick'] is False and obs[0]['can_kick'] is True:
        reward += 0.02
        print_info('can_kick')

    if obs_pre[0]['board'][position_new] == constants.Item.ExtraBomb.value:
        reward += 0.01
        print_info('extrabomb')

    if obs_pre[0]['blast_strength'] < obs[0]['blast_strength']:
        reward += 0.01
        print_info('blast_strength')

    # We are playing a team game.
    if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
        # Team [0, 2] wins.
        return [1+reward, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
        # Team [1, 3] wins.
        return [-1+reward, 1, -1, 1]
    elif step_count >= max_steps:
        # Game is over by max_steps. All agents tie.
        return [-1+reward] * 4
    elif len(alive_agents) == 0:
        # Everyone's dead. All agents tie.
        return [-1+reward] * 4
    else:
        # No team has yet won or lost.
        return [0+reward] * 4