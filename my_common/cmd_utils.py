import argparse

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 通用参数
    parser.add_argument('--model_type', help="PPO2 or dqn", default='ppo')

    # PPO2
    parser.add_argument('--policy_type', help="ResNet or PGN", default='resnet')

    # pretrain
    parser.add_argument('--expert_path', help='专家数据路径', default=None, type=str)
    parser.add_argument('--pretrain_version', help='模式', default='v0', type=str)

    # learn
    parser.add_argument('--env', help='环境名称', type=str, default='PommeRadioCompetition-v2')
    parser.add_argument('--num_env', help='并行环境数量，默认根据 CPU 来选取数量', default=None, type=int)
    parser.add_argument('--num_timesteps', help='强化学习算法训练所用步数', type=float, default=1e6)
    parser.add_argument('--nsteps', type=int, default=128)
    parser.add_argument('--save_interval', help='保存间隔', type=float, default=1e5)

    # play or evaluate
    parser.add_argument('--model0_path', help='加载model0', default=None, type=str)
    parser.add_argument('--model1_path', help='加载model1', default=None, type=str)
    parser.add_argument('--model2_path', help='加载model2', default=None, type=str)
    parser.add_argument('--model3_path', help='加载model3', default=None, type=str)
    parser.add_argument('--using_prune', help='是否使用prune', default=False, action='store_true')

    # save and load
    parser.add_argument('--save_path', help='保存模型', default=None, type=str)
    parser.add_argument('--load_path', help='加载模型', default=None, type=str)
    parser.add_argument('--log_path', help='日志', default=None, type=str)
    parser.add_argument('--using_pgn', help='是否使用pgn', default=False, action='store_true')

    return parser
