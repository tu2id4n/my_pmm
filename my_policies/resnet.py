import tensorflow as tf
import numpy as np
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

def resnet_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu

    layer_1 = conv(scaled_images, 'c1', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME',
                   **kwargs)
    bn1 = tf.layers.batch_normalization(layer_1, name='bn1', training=False)
    res_output = activ(bn1)

    for index, layer_size in enumerate([256] * 20):  # 20层 残差网络
        res_output = res_block(res_output, res_scope='r' + str(index), num=layer_size, **kwargs)

    layer_3 = conv_to_fc(res_output)

    return activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))


def res_block(input, res_scope=None, num=None, **kwargs):
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


class ResNetPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, old_params=None, **kwargs):
        super(ResNetPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                           scale=True)
        print("ResNet")
        with tf.variable_scope('model', reuse=reuse):
            """CNN提取后的特征"""
            extracted_features = resnet_cnn(self.processed_obs, **kwargs)  # 使用残差网络
            extracted_features = tf.layers.flatten(extracted_features)

            pi_h = extracted_features
            pi_latent = pi_h

            vf_h = extracted_features
            vf_latent = vf_h

            value_fn = linear(vf_h, 'vf', n_hidden=1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

            self._value_fn = value_fn
            self._setup_init()

    # TODO: 选取 deterministic 观察效果
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
