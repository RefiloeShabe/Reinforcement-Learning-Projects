import gym
import numpy as np
from reinforce_keras import Agent
from utils import plotLearning


if __name__ == '__main__':
    agent = Agent(ALPHA=0.0005, input_dims=8, GAMMA=0.99, n_actions=4,
                  layer1_size=64, layer2_size=64)
    env = gym.make('LunarLander-v2')
    score_history = []

    n_episodes = 20

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.reset(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)

        agent.learn()

        print('episode ', i, 'score %.1f' % score,
              'average_score %.1f' % np.mean(score_history[-100:]))

        filename = 'Lunar_lander.png'
        plotLearning(score_history, filename=filename, window=100)


