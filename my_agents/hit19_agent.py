import tensorflow as tf
import joblib
from pommerman.agents import *
import numpy as np
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc


class hit19Agent(BaseAgent):
    def __init__(self, pos):
        super(hit19Agent, self).__init__()

        sess = tf.InteractiveSession()
        params = joblib.load('models/hit19')
        self.act_dim = 6
        self.sess = sess
        self.name = 'nets'+pos

        with tf.variable_scope(self.name):
            activ = tf.nn.relu

            self.available_moves = tf.placeholder(tf.float32, [None, self.act_dim], name='availableActions'+pos)
            self.X_ob = tf.placeholder(tf.float32, [None, 11, 11, 30], name="input"+pos)

            layer_1 = conv(self.X_ob, 'c1', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2),
                           pad='SAME',)
            bn1 = tf.layers.batch_normalization(layer_1, name='bn1', training=False)
            res_output = activ(bn1)

            for index, layer_size in enumerate([256] * 20):  # 20层 残差网络
                res_output = self.res(res_output, res_scope='r' + str(index), num=layer_size)

            layer_3 = conv_to_fc(res_output)
            extracted_features = activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))

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

    def res(self, input, res_scope=None, num=None, **kwargs):
        activ = tf.nn.relu

        output_temp = conv(input, res_scope + '_temp', n_filters=num, filter_size=3, stride=1, init_scale=np.sqrt(2),
                           pad='SAME',
                           **kwargs)
        output_temp = tf.layers.batch_normalization(output_temp, name=res_scope + '_temp_bn', training=False)
        output_temp = activ(output_temp)

        output = conv(output_temp, res_scope, n_filters=num, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME',
                      **kwargs)
        output = tf.layers.batch_normalization(output, name=res_scope + '_bn', training=False)
        output = tf.add(output, input)
        output = activ(output)
        return output