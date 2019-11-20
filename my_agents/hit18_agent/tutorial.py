'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import tensorflow as tf
import joblib
from . import convNetwork


def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    ##############
    sess = tf.InteractiveSession()
    params = joblib.load('parametersepoch28')

    # Create a set of agents (exactly four)
    agent_list = [
        convNetwork(sess, params),
        # agents.DockerAgent("pommerman/brain-agent", port=12345),
        # convNetwork(sess, params),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=10080),

        # agents.PlayerAgent(),
        # TODO 建立一个镜像
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    env._max_steps = 500
    # Run the episodes just like OpenAI Gym
    for i_episode in range(100):
        state = env.reset()
        done = False
        while not done:
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            env.render()
        print(done)
        print('Episode {} finished'.format(i_episode))
        print(reward)
    env.close()


if __name__ == '__main__':
    main()
