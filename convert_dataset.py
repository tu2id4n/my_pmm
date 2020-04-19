#  --expert_path=dataset/hako_v1/
from my_common.cmd_utils import arg_parser
import sys
import numpy as np
import time
from tqdm import tqdm
from my_common import get_act_abs, featurize
import _thread
import os


def _convert_dataset(f_name):
    start = time.time()
    dir_path = 'dataset/hako_v1/'
    f_path = dir_path + f_name
    sub_data = np.load(f_path, allow_pickle=True)
    obs = sub_data['obs']
    actions = sub_data['actions']
    del sub_data
    end = time.time()
    print('read file', args.expert_path, end - start)

    start = time.time()
    print("get act_abs")
    for i in tqdm(range(len(obs))):
        # print('action', actions[i])
        actions[i] = get_act_abs(obs[i], actions[i])
        # print('action_abs', actions[i])
    actions = np.array(actions)
    actions = actions.reshape(-1, 1)
    end = time.time()
    print("get act_abs time: ", end - start)

    ob = []
    print('featurize')
    for i in tqdm(range(len(obs))):
        o = featurize(obs[i])
        ob.append(o)
    end = time.time()
    print("featurize time: ", end - start)
    obs = np.array(ob)
    del ob

    print('obs.shape = ', obs.shape)
    print('actions.shape = ', actions.shape)

    numpy_dict = {
        'obs': obs,
        'actions': actions
    }

    np.savez('dataset/hako_v2/' + f_name, **numpy_dict)


if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    files_list = os.listdir(args.expert_path)
    for f_name in files_list:
        _thread.start_new_thread(_convert_dataset, f_name)
