'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
from my_agents import *
from my_common import feature_utils


def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        # RandAgent(),
        # SuicideAgent(),
        agents.PlayerAgent(agent_control="wasd"),  # W,A,S,D to move, E to lay bomb
        StopAgent(),
        StopAgent(),
        # StopAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        agents.PlayerAgent(agent_control="arrows"),  # arrows to move, space to lay bomb
        # SuicideAgent(),
        # SuicideAgent(),
        # SuicideAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(100):
        print('Start to reset')
        state = env.reset()
        print('Reset complete')
        done = False
        while not done:
            actions = env.act(state)
            # print(actions[0])
            # actions[0] = [actions[0], 1, 1]
            state, reward, done, info = env.step(actions)
            bomb_life = state[0]['bomb_life']
            bomb_strenth = state[0]['bomb_blast_strength']
            bomb_life = feature_utils.get_bomb_life(state[0])
            # obs = featurize(state[0], env.position_trav)
            env.render()
            # print(reward)
            # print()
        print(info)
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
