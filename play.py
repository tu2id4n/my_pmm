import sys
import utils
from my_common import feature_utils
from my_common import cmd_utils

def _play():
    print('----------------------------------------------')
    print('|                  P L A Y                   |')
    print('----------------------------------------------')
    env_id = 'PommeRadioCompetition-v4'
    env = utils.make_env(env_id)

    model_type = 'ppo'
    vb = False
    pretrain = False
    model_path0 = 'models/test/v6_30k.zip'
    # model_path0 = 'models/pretrain_v3/pgn_v3_e79.zip'
    model_path1 = None
    model_path2 = None
    model_path3 = None
    model_paths = [model_path0, model_path1, model_path2, model_path3]
    models = utils.get_load_models(model_type, model_paths)

    using_prune = False
    if using_prune:
        prune_agnets = [0, 1, 2, 3]  # 哪些智能体使用剪枝
        nokicks = [True, True, True, True]  # 调整是否使用kick
        print('prune_agents = ', prune_agnets)
        print('nokicks', nokicks)

    for episode in range(100):
        obs = env.reset()
        done = False
        prev2s = [(None, None)] * 4
        total_reward = 0
        while not done:
            all_actions = env.act(obs)

            # Use model
            for i in range(len(models)):
                if models[i] is not None:
                    feature_obs = feature_utils.featurize(obs[i])  # , env.position_trav)
                    if pretrain:
                        action, _states = models[i].predict(feature_obs)
                    else:
                        action_abs, _states = models[i].predict(feature_obs)
                        goal_abs = feature_utils.extra_goal(action_abs, obs[i])
                        utils.print_info('action_obs', action_abs, vb)
                        utils.print_info('goal_obs', goal_abs, vb)
                        # action = _djikstra_act(obs[i], action_abs)
                        action = action_abs
                    if type(action) == list:
                        action = action[0]
                    # print_info('action', action, vb)
                    # print_info('model' + str(i) + ' action: ', action, vb)
                    # if action == 3:
                    #     action = random.randint(0, 5)
                    all_actions[i] = action
                    # print_info('model' + str(i) + ' action: ', action)

            # Use prune
            if using_prune:
                for i in prune_agnets:
                    all_actions[i] = utils.get_modify_act(obs[i], all_actions[i]) #, prev2s[i], nokick=nokicks[i])
                    prev2s[i] = utils.get_prev2obs(prev2s[i], obs[i])

            # 修正为适应通信的动作
            # if args.env == 'PommeRadioCompetition-v2':
            for i in range(len(all_actions)):
                all_actions[i] = [all_actions[i], 1, 1]
            obs, rewards, done, info = env.step(all_actions)
            env.render()
            total_reward += rewards[0]
            if not env._agents[0].is_alive:
                done = True
            # print(all_actions[0])
            # print('reward', rew)
            # print()
        print(info)
        print('total_reward', total_reward)
    env.close()


if __name__ == '__main__':
    arg_parser = cmd_utils.arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _play()
