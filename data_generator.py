import pommerman
from pommerman import agents
from tqdm import tqdm
import sys
from my_common.cmd_utils import arg_parser
import pickle
import random
import time
import numpy as np
from tqdm import tqdm


def _data_generate():
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent()
    ]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    fw = open(args.save_path, 'wb')
    for _ in tqdm(range(10000)):
        simple_list = []
        obs = env.reset()
        done = False
        while not done:
            all_actions = env.act(obs)
            new_obs, rewards, done, info = env.step(all_actions)
            simple_list.append((obs, all_actions, rewards))
            obs = new_obs
        simple_list.append(obs)
        pickle.dump(simple_list, fw)
    env.close()
    fw.close()


def _data_extra():
    fr = open(args.load_path, 'rb')
    obs_list = []
    act_list = []
    pbar = tqdm(total=10000)
    N = 15
    start = time.time()
    while True:
        try:
            pbar.update(1)
            trs = pickle.load(fr)
            if len(trs) < N+1:
                continue
            rand = random.sample(range(0, len(trs) - 1), N)
            # 最后一帧
            obs_t = trs[-1]
            for i in rand:
                obs, acts, _ = trs[i]
                if i == len(trs) - 2:
                    obs_new = obs_t
                else:
                    obs_new, _, _ = trs[i + 1]
                # 如果智能体下一帧还活着，说明做的不是错误动作
                for id in obs_new[0]['alive']:
                    id = id - 10
                    obs_list.append(obs[id])
                    act_list.append(acts[id])
        except:
            print(args.load_path, 'extra ok!')
            break
    fr.close()
    end = time.time()
    print('use time:', end - start)
    numpy_dict = {
        'obs': obs_list,
        'actions': act_list
    }
    for key, val in numpy_dict.items():
        print(key, np.array(val).shape, type(val))
    np.savez(args.save_path + '.npz', **numpy_dict)


if __name__ == '__main__':
    arg_parser = arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    if args.version == 'v0':
        _data_generate()
    elif args.version == 'v1':
        _data_extra()
