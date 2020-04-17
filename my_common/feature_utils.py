import numpy as np
from gym import spaces
import copy
import random
from .prune import get_filtered_actions
from pommerman import utility, constants
from collections import defaultdict
import queue
from my_pommerman import position_is_passable


def get_observertion_space():
    return spaces.Box(low=0, high=1, shape=(11, 11, 24))


def get_action_space():
    return spaces.Discrete(122)


def get_pre_action_space():
    return spaces.Discrete(6)


def featurize(obs_nf, position_trav, action_pre=None):
    return _featurize1(obs_nf, position_trav, action_pre)  # 11 * 11 * 24
    # return _featurize2(obs_nf)  # 11 * 11 * 30


# 提取goal_abs:
def extra_goal(goal_abs, obs=None):
    if goal_abs == 121:
        return obs['postion']
    for r in range(0, 11):
        for c in range(0, 11):
            if r * 11 + c == goal_abs:
                return (r, c)


def _djikstra_act(obs_nf, goal_abs, exclude=None):
    if goal_abs == 121:
        return 5

    board = np.array(obs_nf['board'])
    my_position = tuple(obs_nf['position'])
    enemies = [constants.Item(e) for e in obs_nf['enemies']]

    if exclude is None:
        exclude = [
            constants.Item.Rigid,
            constants.Item.Flames,
            constants.Item.Wood,
            constants.Item.Bomb,
        ]

    dist = {}
    prev = {}
    Q = queue.Queue()

    # my_x, my_y = my_position
    for r in range(0, 11):
        for c in range(0, 11):
            position = (r, c)

            if any([utility.position_in_items(board, position, exclude)]):
                continue

            prev[position] = None

            if position == my_position:
                Q.put(position)
                dist[position] = 0
            else:
                dist[position] = np.inf

    while not Q.empty():
        position = Q.get()

        if position_is_passable(board, position, enemies):
            x, y = position
            val = dist[(x, y)] + 1
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (row + x, col + y)
                if new_position not in dist:
                    continue

                if val < dist[new_position]:
                    dist[new_position] = val
                    prev[new_position] = position
                    Q.put(new_position)
                elif val == dist[new_position] and random.random() < .5:
                    dist[new_position] = val
                    prev[new_position] = position

    goal = extra_goal(goal_abs)
    while goal in dist and prev[goal] != my_position:
        goal = prev[goal]

    # up, down, left, right
    my_x, my_y = my_position
    count = 1
    for act_abs in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        row, col = act_abs
        if goal == (my_x + row, my_y + col):
            return count
        count += 1
    return 0


def _featurize1(obs_nf, position_trav, action_pre):
    obs = copy.deepcopy(obs_nf)
    board = np.array(obs['board'])

    maps = []
    """棋盘物体 one-hot"""
    for i in range(9):  # [0, 1, ..., 8]  9
        maps.append(board == i)
    maps.append(np.array(obs['bomb_blast_strength']) / 10)  # 1 最大可能为23
    maps.append(np.array(obs['bomb_life'] / 9))  # 1 最大为9

    bomb_direction = np.array(obs['bomb_moving_direction'])  # 四个方向
    for i in range(1, 5):
        maps.append(bomb_direction == i)  # 4

    maps.append(np.array(obs['flame_life']) / 3)  # 1 最大为3

    """一个队友的位置 one-hot """
    teammate_idx = obs['teammate'].value
    if not teammate_idx == 9:  # AgentDummy
        maps.append(board == teammate_idx)  # 1

    """两个敌人的位置 one-hot"""
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
    maps.append(np.logical_or(board == enemies_idx[0], board == enemies_idx[1]))  # 1

    """训练智能体的位置 one-hot"""
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    maps.append(board == train_agent_idx)  # 1

    maps.append(np.full(board.shape, obs['ammo']) / 10)
    maps.append(np.full(board.shape, obs['blast_strength']) / 10)
    maps.append(np.full(board.shape, obs['can_kick']))

    '''获得访问过的positions'''
    position_map = np.zeros(shape=(11, 11))
    for p in position_trav:
        position_map[p] = 1
    maps.append(position_map)

    '''加入上一次的goal'''
    goal_map = np.zeros(shape=(11, 11))
    if action_pre is None:
        goal_map[obs['position']] = 1
    else:
        goal_map[extra_goal(action_pre, obs)] = 1
    maps.append(goal_map)

    return np.stack(maps, axis=2)  # 11*11*24


def scalar_feature(obs1):
    obs = copy.deepcopy(obs1)
    maps = [obs['blast_strength'], int(obs['can_kick']), obs['ammo']]

    return maps  # 3


# 11*11*30
def _featurize2(obs_nf):
    obs = copy.deepcopy(obs_nf)
    board = np.array(obs['board'])

    maps = []
    """棋盘物体 one-hot"""
    for i in range(9):
        maps.append(board == i)  # --> 9

    '''爆炸威胁 one-hot'''
    bomb_life = get_bomb_life(obs)
    for i in range(2, 13):
        maps.append(bomb_life == i)  # --> 11

    '''bomb_direction one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction'].copy()
    bomb_moving_direction = np.array(bomb_moving_direction)
    for i in range(1, 5):
        maps.append(bomb_moving_direction == i)  # --> 4

    """标量映射为11*11的矩阵"""
    maps.append(np.full(board.shape, obs['ammo'] / 3))  # --> 1
    maps.append(np.full(board.shape, obs['blast_strength'] / 5))  # --> 1
    maps.append(np.full(board.shape, obs['can_kick']))  # --> 1

    """一个队友的位置 one-hot """
    teammate_idx = obs['teammate'].value
    if not teammate_idx == 9:  # AgentDummy
        maps.append(board == teammate_idx)  # --> 1

    """两个敌人的位置 one-hot"""
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
    maps.append(np.logical_or(board == enemies_idx[0], board == enemies_idx[1]))  # --> 1

    """训练智能体的位置 one-hot"""
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    maps.append(board == train_agent_idx)  # --> 1

    return np.stack(maps, axis=2)  # 11*11*30


def get_bomb_life(obs1):
    obs = copy.deepcopy(obs1)
    board = np.array(obs['board'])
    bomb_life = np.array(obs['bomb_life'])
    bomb_blast_strength = np.array(obs['bomb_blast_strength'])
    flame_life = np.array(obs['flame_life'])

    # 统一炸弹时间
    for x in range(11):
        for y in range(11):
            if bomb_blast_strength[(x, y)] > 0:
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x - i, y)
                    if x - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y + i)
                    if y + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y - i)
                    if y - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]

    bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
    flame_life = np.where(flame_life == 0, 15, flame_life)
    flame_life = np.where(flame_life == 1, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)

    return bomb_life


def get_modify_act(obs, act, prev, info=False, nokick=True):
    valid_actions = get_filtered_actions(obs.copy(), prev_two_obs=prev, nokick=nokick)
    if act not in valid_actions:
        if info:
            print(act)
            print(valid_actions)
        act = random.sample(valid_actions, 1)
    if type(act) == list:
        act = act[0]
    return act


def get_prev2obs(prev, obs):
    old_old, old = prev
    old_old = copy.deepcopy(old)
    old = copy.deepcopy(obs)
    new_prev = (old_old, old)
    return new_prev
