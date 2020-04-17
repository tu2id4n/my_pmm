import sys
from utils import *
from my_common import *
from my_common import _djikstra_act


def _play():
    env = make_env(args.env)
    model_paths = [args.model0_path, args.model1_path, args.model2_path, args.model3_path]
    models = get_load_models(args.model_type, model_paths, args.log_path, using_pgn=args.using_pgn)
    print(
        "**************** Start to play ****************************************************************")

    if args.using_prune:
        using_prune = [0, 1, 2, 3]  # 哪些智能体使用剪枝
        nokicks = [False] * 4  # 调整是否使用kick
        print('using_prune = ', using_prune)
        print('nokicks', nokicks)

    for episode in range(100):
        obs = env.reset()
        done = False
        prev2s = [(None, None)] * 4
        rew = 0
        while not done:
            all_actions = env.act(obs)

            # Use model
            for i in range(len(models)):
                if models[i] is not None:
                    feature_obs = featurize(obs[i], env.position_trav)
                    action_abs, _states = models[i].predict(feature_obs)
                    goal_abs = extra_goal(action_abs, obs[i])
                    print('action_obs', action_abs)
                    print('goal_obs', goal_abs)
                    action = _djikstra_act(obs[i], action_abs)
                    if type(action) == list:
                        action = action[0]
                    print('action', action)
                    # print('model', i, ' action: ', action)
                    # if action == 3:
                    #     action = random.randint(0, 5)
                    all_actions[i] = action
                    # print('model', i, ' action: ', action)
                    # nokicks[i] = False

            # Use prune
            if args.using_prune:
                for i in using_prune:
                    all_actions[i] = get_modify_act(obs[i], all_actions[i], prev2s[i], nokick=nokicks[i])
                    prev2s[i] = get_prev2obs(prev2s[i], obs[i])

            # 修正为适应通信的动作
            if args.env == 'PommeRadioCompetition-v2':
                for i in range(len(all_actions)):
                    all_actions[i] = [all_actions[i], 1, 1]

            obs, rewards, done, info = env.step(all_actions)
            env.render()
            rew += rewards[0]
            print(all_actions[0])
            print('reward', rew)
            print()
        print(info)
        print('reward', rew)
    env.close()


if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    _play()
