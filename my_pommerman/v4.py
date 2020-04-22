"""The Pommerman v2 Environment, which has communication across the agents.

The communication works by allowing each agent to send a vector of
radio_num_words (default = 2) from a vocabulary of size radio_vocab_size
(default = 8) to its teammate each turn. These vectors are passed into the
observation stream for each agent.
"""
from gym import spaces
import numpy as np

from pommerman import constants
from pommerman import utility
from pommerman.envs import v0
from my_pommerman import make_board_v3, make_items_v3
from my_pommerman import reward_shaping
import copy
from my_common import feature_utils


class Pomme(v0.Pomme):
    '''The hardest pommerman environment. This class expands env v0
    adding communication between agents.'''
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
        'video.frames_per_second': constants.RENDER_FPS
    }
    def __init__(self, *args, **kwargs):
        self._radio_vocab_size = kwargs.get('radio_vocab_size')
        self._radio_num_words = kwargs.get('radio_num_words')
        if (self._radio_vocab_size and
                not self._radio_num_words) or (not self._radio_vocab_size and
                                               self._radio_num_words):
            assert ("Include both radio_vocab_size and radio_num_words.")

        self._radio_from_agent = {
            agent: (0, 0)
            for agent in [
                constants.Item.Agent0, constants.Item.Agent1,
                constants.Item.Agent2, constants.Item.Agent3
            ]
        }
        super().__init__(*args, **kwargs)

    def _set_action_space(self):
        self.action_space = spaces.Tuple(
            tuple([spaces.Discrete(6)] +
                  [spaces.Discrete(self._radio_vocab_size
                                  )] * self._radio_num_words))

    def _set_observation_space(self):
        """The Observation Space for each agent.

        Total observatiosn: 3*board_size^2 + 12 + radio_vocab_size * radio_num_words:
        - all of the board (board_size^2)
        - bomb blast strength (board_size^2).
        - bomb life (board_size^2)
        - agent's position (2)
        - player ammo counts (1)
        - blast strength (1)
        - can_kick (1)
        - teammate (one of {AgentDummy.value, Agent3.value}).
        - enemies (three of {AgentDummy.value, Agent3.value}).
        - radio (radio_vocab_size * radio_num_words)
        """
        bss = self._board_size**2
        min_obs = [0] * 3 * bss + [0] * 5 + [constants.Item.AgentDummy.value
                                            ] * 4
        max_obs = [len(constants.Item)] * bss + [self._board_size
                                                ] * bss + [25] * bss
        max_obs += [self._board_size] * 2 + [self._num_items] * 2 + [1]
        max_obs += [constants.Item.Agent3.value] * 4
        min_obs.extend([0] * self._radio_vocab_size * self._radio_num_words)
        max_obs.extend([1] * self._radio_vocab_size * self._radio_num_words)
        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def get_observations(self):
        observations = super().get_observations()
        for obs in observations:
            obs['message'] = self._radio_from_agent[obs['teammate']]

        self.observations = observations
        return observations

    def make_board(self):
        self._board = make_board_v3(self._board_size, self._num_rigid,
                                         self._num_wood, len(self._agents))
    def make_items(self):
        self._items = make_items_v3(self._board, self._num_items)

    def reset(self):
        assert (self._agents is not None)
        self.act_abs_pre = None
        if self._init_game_state is not None:
            self.set_json_info()
        else:
            self._step_count = 0
            self.make_board()
            self.make_items()
            self._bombs = []
            self._flames = []
            self._powerups = []
            for agent_id, agent in enumerate(self._agents):
                pos = np.where(self._board == utility.agent_value(agent_id))
                row = pos[0][0]
                col = pos[1][0]
                agent.set_start_position((row, col))
                agent.reset()
        return self.get_observations()

    def step(self, actions):
        self.act_abs_pre = actions[0][0]
        self.obs_pre = copy.deepcopy(self.get_observations())
        actions[0][0] = feature_utils._djikstra_act(self.obs_pre[0], self.act_abs_pre)

        personal_actions = []
        radio_actions = []
        for agent_actions, agent in zip(actions, self._agents):
            if type(agent_actions) == int or not agent.is_alive:
                personal_actions.append(agent_actions)
                radio_actions.append((0, 0))
            elif type(agent_actions) in [tuple, list]:
                personal_actions.append(agent_actions[0])
                radio_actions.append(
                    tuple(agent_actions[1:(1+self._radio_num_words)]))
            else:
                raise

            self._radio_from_agent[getattr(
                constants.Item, 'Agent%d' % agent.agent_id)] = radio_actions[-1]

        # return super().step(personal_actions)
        self._intended_actions = personal_actions
        actions = personal_actions
        max_blast_strength = self._agent_view_size or 10
        result = self.model.step(
            actions,
            self._board,
            self._agents,
            self._bombs,
            self._items,
            self._flames,
            max_blast_strength=max_blast_strength)
        self._board, self._agents, self._bombs, self._items, self._flames = \
            result[:5]

        done = self._get_done()
        obs = self.get_observations()
        reward = self._get_rewards()
        info = self._get_info(done, reward)
        if done:
            # Callback to let the agents know that the game has ended.
            for agent in self._agents:
                agent.episode_end(reward[agent.agent_id])

        self._step_count += 1
        return obs, reward, done, info

    def _get_rewards(self):
        return reward_shaping.get_rewards_v3_7(self._agents, self._step_count, self._max_steps, self.obs_pre, self.get_observations(), self.act_abs_pre)

    @staticmethod
    def featurize(obs):
        ret = super().featurize(obs)
        message = obs['message']
        message = utility.make_np_float(message)
        return np.concatenate((ret, message))

    def get_json_info(self):
        ret = super().get_json_info()
        ret['radio_vocab_size'] = json.dumps(
            self._radio_vocab_size, cls=json_encoder)
        ret['radio_num_words'] = json.dumps(
            self._radio_num_words, cls=json_encoder)
        ret['_radio_from_agent'] = json.dumps(
            self._radio_from_agent, cls=json_encoder)
        return ret

    def set_json_info(self):
        super().set_json_info()
        self.radio_vocab_size = json.loads(
            self._init_game_state['radio_vocab_size'])
        self.radio_num_words = json.loads(
            self._init_game_state['radio_num_words'])
        self._radio_from_agent = json.loads(
            self._init_game_state['_radio_from_agent'])
