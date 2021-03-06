import numpy as np
from gym import spaces
import copy
import random
from .prune import get_filtered_actions
from pommerman import utility, constants
from collections import defaultdict
import queue
from my_pommerman import position_is_passable


def print_info(info, vb=False):
    if vb:
        print(info)


def get_observertion_space():
    # return spaces.Box(low=0, high=1, shape=(11, 11, 30))
    return spaces.Box(low=0, high=1, shape=(8, 8, 26))


def get_action_space():
    # return spaces.Discrete(122)
    return spaces.Discrete(65)


def featurize(obs_nf, position_trav=set(), action_pre=None):
    # return _featurize1(obs_nf, position_trav, action_pre)  # 11 * 11 * 24
    # return _featurize2(obs_nf)  # 11 * 11 * 30
    return _featurize_8m8(obs_nf)  # 25*8*8


def _djikstra_act(obs_nf, goal_abs, rang=11):
    # return _djikstra_act_v3(obs_nf, goal_abs, rang=rang)
    return _djikstra_act_v4(obs_nf, goal_abs, rang=rang)


# 提取goal_abs:
def extra_goal(goal_abs, obs=None, rang=11):
    if goal_abs == rang * rang:
        return obs['position']
    for r in range(0, rang):
        for c in range(0, rang):
            if r * rang + c == goal_abs:
                return (r, c)


def extra_position(item, board):
    for r in range(0, 11):
        for c in range(0, 11):
            if board[(r, c)] == item:
                return (r, c)
    return None


# up, down, left, right
# [(-1, 0), (1, 0), (0, -1), (0, 1)]:
def get_my_bomb_life(bomb_life, my_position, rang=11):
    q = queue.Queue()
    q.put(my_position)
    used_position = []
    my_bomb_life = np.zeros(shape=(rang, rang))
    current_bomb_life = bomb_life[my_position]
    if current_bomb_life > 0:
        while not q.empty():
            position = q.get()
            my_r, my_c = position
            for act in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                r = act[0] + my_r
                c = act[1] + my_c
                if 0 <= r <= rang - 1 and 0 <= c <= rang - 1 and (r, c) not in used_position:
                    if bomb_life[(r, c)] == current_bomb_life:
                        q.put((r, c))
                        my_bomb_life[(r, c)] = bomb_life[(r, c)]
                        used_position.append((r, c))
    return my_bomb_life


def _djikstra_act_v1(obs_nf, goal_abs, exclude=None):
    if goal_abs == 121:
        return 5

    board = np.array(obs_nf['board'])
    my_position = tuple(obs_nf['position'])
    enemies = [constants.Item(e) for e in obs_nf['enemies']]

    if exclude is None:
        exclude = [
            constants.Item.Rigid,
            constants.Item.Wood,
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


def _djikstra_act_v2(obs_nf, goal_abs, exclude=None, rang=11):
    # 放炸弹
    if goal_abs == rang * rang:
        print_info('释放炸弹')
        return 5

    # 停止在原地
    my_position = tuple(obs_nf['position'])
    goal = extra_goal(goal_abs, rang=rang)
    if goal == my_position:
        print_info('停在原地')
        return 0

    board = np.array(obs_nf['board'])
    enemies = [constants.Item(e) for e in obs_nf['enemies']]

    if exclude is None:
        exclude = [
            constants.Item.Rigid,
            constants.Item.Wood,
        ]

    dist = {}
    prev = {}
    Q = queue.Queue()

    # my_x, my_y = my_position
    for r in range(0, rang):
        for c in range(0, rang):
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

    row_g, col_g = goal
    my_x, my_y = my_position
    up = (-1, 0)
    down = (1, 0)
    left = (0, -1)
    right = (0, 1)
    # 判断goal是否可以达到
    while goal in dist and prev[goal] != my_position:
        goal = prev[goal]

    legal_act = []
    # 无法到达有效目的
    if goal not in dist:
        # 可以向下行走
        if row_g > my_x:
            if isLegal_act(obs_nf, down, rang=rang): legal_act.append(2)
        elif row_g < my_x:
            if isLegal_act(obs_nf, up, rang=rang): legal_act.append(1)
        # 可以向右行走
        if col_g > my_x:
            if isLegal_act(obs_nf, right, rang=rang): legal_act.append(4)
        elif col_g < my_x:
            if isLegal_act(obs_nf, left, rang=rang): legal_act.append(3)
        if legal_act:
            print_info('无法到达目的地，但是向此方向移动')
            return random.choice(legal_act)
    # 可以达到的目的
    else:
        count = 1
        for act_to in [up, down, left, right]:
            row, col = act_to
            if goal == (my_x + row, my_y + col):
                print_info('向目的地移动')
                return count
            count += 1
    print_info('非法的移动动作')
    return 0


def _djikstra_act_v3(obs_nf, goal_abs, exclude=None, rang=11):
    # 放炸弹
    if goal_abs == rang * rang:
        print_info('释放炸弹')
        return 5

    # 停止在原地
    my_position = tuple(obs_nf['position'])
    goal = extra_goal(goal_abs, rang=rang)
    if goal == my_position:
        print_info('停在原地')
        return 0

    board = np.array(copy.deepcopy(obs_nf['board']))

    enemies = [constants.Item(e) for e in obs_nf['enemies']]

    if exclude is None:
        exclude = [
            constants.Item.Rigid,
            constants.Item.Wood,
        ]

    # 判断是否合法
    if board[goal] in exclude:
        goal_x, goal_y = goal
        if 0 <= goal_x-1 < rang and board[(goal_x - 1, goal_y)] not in exclude:
            goal = (goal_x - 1, goal_y)
        elif 0 <= goal_x+1 < rang and board[(goal_x + 1, goal_y)] not in exclude:
            goal = (goal_x + 1, goal_y)
        elif 0 <= goal_y-1 < rang and board[(goal_x, goal_y-1)] not in exclude:
            goal = (goal_x, goal_y-1)
        elif 0 <= goal_y+1 < rang and board[(goal_x, goal_y+1)] not in exclude:
            goal = (goal_x, goal_y+1)

    dist = {}
    prev = {}
    Q = queue.Queue()

    # my_x, my_y = my_position
    for r in range(0, rang):
        for c in range(0, rang):
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

    row_g, col_g = goal
    my_x, my_y = my_position
    up = (-1, 0)
    down = (1, 0)
    left = (0, -1)
    right = (0, 1)
    # 判断goal是否可以达到
    while goal in dist and prev[goal] != my_position:
        goal = prev[goal]

    legal_act = []
    # 无法到达有效目的
    if goal not in dist:
        # 可以向下行走
        if row_g > my_x:
            if isLegal_act(obs_nf, down, rang=rang): legal_act.append(2)
        elif row_g < my_x:
            if isLegal_act(obs_nf, up, rang=rang): legal_act.append(1)
        # 可以向右行走
        if col_g > my_x:
            if isLegal_act(obs_nf, right, rang=rang): legal_act.append(4)
        elif col_g < my_x:
            if isLegal_act(obs_nf, left, rang=rang): legal_act.append(3)
        if legal_act:
            print_info('无法到达目的地，但是向此方向移动')
            return random.choice(legal_act)
    # 可以达到的目的
    else:
        count = 1
        for act_to in [up, down, left, right]:
            row, col = act_to
            if goal == (my_x + row, my_y + col):
                print_info('向目的地移动')
                return count
            count += 1
    print_info('非法的移动动作')
    return 0

def _djikstra_act_v4(obs_nf, goal_abs, exclude=None, rang=11):
    # 放炸弹
    if goal_abs == rang * rang:
        print_info('释放炸弹')
        return 5

    # 停止在原地
    my_position = tuple(obs_nf['position'])
    goal = extra_goal(goal_abs, rang=rang)
    # print('goal', goal)
    if goal == my_position:
        print_info('停在原地')
        return 0

    board = np.array(obs_nf['board'])
    enemies = [constants.Item(e) for e in obs_nf['enemies']]

    if exclude is None:
        exclude = [
            constants.Item.Rigid,
            constants.Item.Wood,
        ]

    dist = {}
    prev = {}
    Q = queue.Queue()

    # my_x, my_y = my_position
    for r in range(0, rang):
        for c in range(0, rang):
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

    row_g, col_g = goal
    my_x, my_y = my_position
    up = (-1, 0)
    down = (1, 0)
    left = (0, -1)
    right = (0, 1)
    # 判断goal是否可以达到
    while goal in dist and prev[goal] != my_position:
        goal = prev[goal]

    legal_act = []
    # 无法到达有效目的
    if goal not in dist:
        # print('random')
        goal_abs = random.randint(0, rang * rang)
        return _djikstra_act_v4(obs_nf, goal_abs, rang=rang)
    # 可以达到的目的
    else:
        count = 1
        for act_to in [up, down, left, right]:
            row, col = act_to
            if goal == (my_x + row, my_y + col):
                print_info('向目的地移动')
                return count
            count += 1
    print_info('非法的移动动作')
    return 0

def isLegal_act(obs_nf, act_to, rang=11):
    my_x, my_y = obs_nf['position']
    row, col = act_to
    passage = constants.Item.Passage.value
    bomb = constants.Item.Passage.value
    if 0 <= my_x + row <= rang - 1 and 0 <= my_y + col <= rang - 1:
        if obs_nf['can_kick']:
            return obs_nf['board'][(my_x + row, my_y + col)] in [bomb, passage]
        else:
            return obs_nf['board'][(my_x + row, my_y + col)] == passage
    else:
        return False


def get_act_abs(obs, action, rang=11):
    if action == 5:
        return rang * rang

    r, c = obs['position']
    act_abs = r * rang + c
    if action == 0:
        return act_abs

    count = 1000
    while count > 0:
        rand_act_obs = random.randint(0, rang * rang-1)
        if _djikstra_act(obs_nf=obs, goal_abs=rand_act_obs, rang=rang) == action:
            return rand_act_obs
        count -= 1

    return act_abs


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
    print(position_map)

    '''加入上一次的goal'''
    goal_map = np.zeros(shape=(11, 11))
    if action_pre is None:
        goal_map[obs['position']] = 1
    else:
        goal_map[extra_goal(action_pre, obs)] = 1
    # maps.append(goal_map)

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
    maps.append(np.full(board.shape, obs['ammo'] / 5))  # --> 1
    maps.append(np.full(board.shape, obs['blast_strength'] / 10))  # --> 1
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


def get_bomb_life(obs_nf, rang=11):
    obs = copy.deepcopy(obs_nf)
    board = np.array(obs['board'])
    bomb_life = np.array(obs['bomb_life'])
    bomb_blast_strength = np.array(obs['bomb_blast_strength'])
    flame_life = np.array(obs['flame_life'])

    # 统一炸弹时间
    for x in range(rang):
        for y in range(rang):
            if bomb_blast_strength[(x, y)] > 0:
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > rang - 1:
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
                    if y + i > rang - 1:
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


def get_modify_act(obs, act, prev=[None, None], info=False, nokick=True):
    valid_actions = get_filtered_actions(obs.copy(), prev_two_obs=prev, nokick=nokick)
    # if 5 in valid_actions:
    #     return 5
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


# 29 8 8
def _featurize_8m8(obs_nf):
    obs = copy.deepcopy(obs_nf)
    board = np.array(obs['board'])

    maps = []
    """棋盘物体 one-hot"""
    for i in range(1, 9):
        maps.append(board == i)  # --> 8

    '''爆炸威胁 one-hot'''
    bomb_life = get_bomb_life(obs, rang=8)
    for i in range(2, 13):
        maps.append(bomb_life == i)  # --> 11

    '''bomb_direction one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction'].copy()
    bomb_moving_direction = np.array(bomb_moving_direction)
    # for i in range(1, 5):
    #     maps.append(bomb_moving_direction == i)  # --> 4
    maps.append(np.full(board.shape, obs['bomb_moving_direction'] / 4))  # --> 1

    """标量映射为11*11的矩阵"""
    maps.append(np.full(board.shape, obs['ammo'] / 5))  # --> 1
    maps.append(np.full(board.shape, obs['blast_strength'] / 10))  # --> 1
    maps.append(np.full(board.shape, obs['can_kick']))  # --> 1

    """敌人的位置 one-hot"""
    maps.append(board == 11)  # 1

    """训练智能体的位置 one-hot"""

    train_agent_idx = 10
    maps.append(board == train_agent_idx)  # 1

    return np.stack(maps, axis=0)  # 25*8*8
