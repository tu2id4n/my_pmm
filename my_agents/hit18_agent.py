import tensorflow as tf
import joblib
from pommerman.agents import *
import numpy as np


def available_action(state):
    # 可行动的动作为0，不可行动的动作为-9999
    position = state['position']
    board = state['board']
    x = position[0]
    y = position[1]
    if state['can_kick']:
        avail_path = [0, 3, 6, 7, 8]
    else:
        avail_path = [0, 6, 7, 8]
    action = [0, -9999, -9999, -9999, -9999, 0]
    if state['ammo'] == 0:
        action[-1] = -9999
    if (x - 1) >= 0:
        if board[x - 1, y] in avail_path:
            # 可以往上边走
            action[1] = 0
    if (x + 1) <= 10:
        if board[x + 1, y] in avail_path:
            # 可以往下边走
            action[2] = 0
    if (y - 1) >= 0:
        if board[x, y - 1] in avail_path:
            # 可以往坐边走
            action[3] = 0
    if (y + 1) <= 10:
        if board[x, y + 1] in avail_path:
            # 可以往右边走
            action[4] = 0
    return action


def featurize(obs):
    # TODO: history of n moves?
    board = obs['board']
    board = np.array(board)

    # convert board items into bitmaps
    maps = [board == i for i in range(10)]
    maps.append(np.array(obs['bomb_blast_strength']))
    maps.append(np.array(obs['bomb_life']))

    # duplicate ammo, blast_strength and can_kick over entire map
    # 创建一个由常数填充的数组,第一个参数是数组的形状，第二个参数是数组中填充的常数。
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add my position as bitmap
    position = np.zeros(board.shape)
    position[tuple(obs['position'])] = 1
    maps.append(position)

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'])
    else:
        maps.append(np.zeros(board.shape))

    # add enemies
    enemies = [board == e for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))

    # assert len(maps) == NUM_CHANNELS
    return np.stack(maps, axis=2)


class hit18Agent(BaseAgent):
    def __init__(self, pos):
        super(hit18Agent, self).__init__()

        sess = tf.InteractiveSession()
        params = joblib.load('models/hit18')
        self.act_dim = 6
        self.sess = sess
        self.name = 'nets'+pos

        with tf.variable_scope(self.name):
            activation = tf.nn.relu
            self.available_moves = tf.placeholder(tf.float32, [None, self.act_dim], name='availableActions'+pos)
            self.X_ob = tf.placeholder(tf.float32, [None, 11, 11, 18], name="input"+pos)

            conv1 = tf.layers.conv2d(inputs=self.X_ob,
                                     filters=256,
                                     kernel_size=3,
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     name='conv1'+pos
                                     )
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=256,
                                     kernel_size=3,
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     name='conv2'+pos
                                     )
            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=256,
                                     kernel_size=3,
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     name='conv3'+pos
                                     )
            feature_dim = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]
            h = tf.reshape(conv3, [-1, feature_dim])
            self.pi = tf.layers.dense(h, self.act_dim, name='p'+pos)
            self.vf = tf.layers.dense(h, 1, activation=tf.nn.tanh, name='v'+pos)

        self.availPi = tf.add(self.pi, self.available_moves)
        # TODO argmax instead of sample
        self.dist = tf.distributions.Categorical(logits=self.pi)
        self.action = self.dist.sample()
        self.neglog_probs = -self.dist.log_prob(self.action)
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.loadParams(params)

    def step_policy(self, obs):
        action = self.sess.run(self.action, {self.X_ob: obs})
        return action

    def act(self, obs, action_space):
        # featurize obs
        obs_input = featurize(obs).reshape(-1, 11, 11, 18)
        # availacs = np.array(available_action(obs),dtype=int).reshape(1,6)
        action = self.step_policy(obs_input)
        action = np.int(action)
        return action

    def loadParams(self, paramsToLoad):
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.params, paramsToLoad)]
        self.sess.run(self.replace_target_op)
