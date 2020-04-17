'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
from my_agents import *

def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeRadioCompetition-v3', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        print('Start to reset')
        state = env.reset()
        print('Reset complete')
        done = False
        while not done:
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            env.render()
            print(reward)
            print()
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
