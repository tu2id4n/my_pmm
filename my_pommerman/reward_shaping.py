from pommerman import constants


def get_rewards_v3(agents, step_count, max_steps, obs_pre, obs, position_trav):
    def any_lst_equal(lst, values):
        '''Checks if list are equal'''
        return any([lst == v for v in values])

    reward = 0
    position_new = obs[0]['position']
    if position_new not in position_trav:
        reward += 0.001

    if obs_pre[0]['can_kick'] is False and obs[0]['can_kick'] is True:
        reward += 0.02

    if obs_pre[0]['ammo'] < obs[0]['ammo']:
        reward += 0.01

    if obs_pre[0]['blast_strength'] < obs[0]['blast_strength']:
        reward += 0.01

    alive_agents = [num for num, agent in enumerate(agents) \
                    if agent.is_alive]

    # We are playing a team game.
    if any_lst_equal(alive_agents, [[0, 2], [0]]):
        # Team [0, 2] wins and agent0 alive
        return [reward + 1, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[2]]):
        # Team [0, 2] wins but agent0 is dead
        return [reward + 0.5, -1, 1, -1]
    elif any_lst_equal(alive_agents, [[1, 3]]):
        # Team [1, 3] wins and no enemy dead
        return [reward - 1, 1, -1, 1]
    elif any_lst_equal(alive_agents, [[1], [3]]):
        # Team [1, 3] wins and one enemy dead
        return [reward - 0.5, 1, -1, 1]
    elif step_count >= max_steps:
        # Game is over by max_steps. All agents tie.
        return [reward] * 4
    elif len(alive_agents) == 0:
        # Everyone's dead. All agents tie.
        return [reward] * 4
    else:
        # No team has yet won or lost.
        return [reward] * 4
