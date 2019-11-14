import numpy as np
from gym import spaces
import copy
import random
from .prune import get_filtered_actions


def get_observertion_space():
    return spaces.Box(low=0, high=1, shape=(11, 11, 30))


def get_action_space():
    return spaces.Discrete(6)


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


# 11*11*30
def featurize(obs1):
    obs = copy.deepcopy(obs1)
    board = np.array(obs['board'])

    maps = []
    """棋盘物体 one-hot"""
    for i in range(9):  # [0, 1, ..., 8] --> 9
        maps.append(board == i)

    '''爆炸威胁 one-hot'''
    bomb_life = get_bomb_life(obs)
    for i in range(2, 13):  # --> 12
        maps.append(bomb_life == i)

    '''bomb_direction one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction'].copy()
    bomb_moving_direction = np.array(bomb_moving_direction)
    for i in range(1, 5):  # --> 4
        maps.append(bomb_moving_direction == i)

    """标量映射为11*11的矩阵"""  # --> 3
    maps.append(np.full(board.shape, obs['ammo'] / 3))
    maps.append(np.full(board.shape, obs['blast_strength'] / 5))
    maps.append(np.full(board.shape, obs['can_kick']))

    """一个队友的位置 one-hot """  # --> 1
    teammate_idx = obs['teammate'].value
    if not teammate_idx == 9:  # AgentDummy
        maps.append(board == teammate_idx)

    """两个敌人的位置 one-hot"""  # --> 2
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
    maps.append(np.logical_or(board == enemies_idx[0], board == enemies_idx[1]))

    """训练智能体的位置 one-hot"""  # --> 1
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    maps.append(board == train_agent_idx)

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
